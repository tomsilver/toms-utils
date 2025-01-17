"""Interfaces and utilities for large language models."""

import abc
import base64
import importlib
import json
import logging
import multiprocessing as mp
import operator
import os
import signal
import string
import sys
import tempfile
import traceback
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Collection

import google.generativeai as genai
import imagehash
import openai
import PIL.Image
from gymnasium.spaces import Box, Space
from tenacity import retry, stop_after_attempt, wait_random_exponential

from tomsutils.utils import consistent_hash

# This speeds up the sandbox for code synthesis by a lot.
mp.set_start_method("fork")

# This is a special string that we assume will never appear in a prompt, and
# which we use to separate prompt and completion in the cache. The reason to
# do it this way, rather than saving the prompt and responses separately,
# is that we want it to be easy to browse the cache as text files.
_CACHE_SEP = "\n####$$$###$$$####$$$$###$$$####$$$###$$$###\n"


class PretrainedLargeModel(abc.ABC):
    """A pretrained large vision or language model."""

    def __init__(self, cache_dir: Path, use_cache_only: bool = False) -> None:
        self._cache_dir = cache_dir
        self._use_cache_only = use_cache_only

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this model.

        This identifier should include sufficient information so that
        querying the same model with the same prompt and same identifier
        should yield the same result (assuming temperature 0).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        """This is the main method that subclasses must implement.

        This helper method is called by sample_completions(), which
        caches the prompts and responses to disk.
        """
        raise NotImplementedError("Override me!")

    def sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.
        The seed may not be used and the results may therefore not be
        reproducible for models where we only have access through an API
        that does not expose the ability to set a random seed. Responses
        are saved to disk.
        """
        cache_filepath = self._get_cache_filepath(
            prompt, imgs, temperature, seed, num_completions, filename="prompt.txt"
        )
        if not cache_filepath.exists():
            if self._use_cache_only:
                raise ValueError("No cached response found for prompt.")
            logging.debug(f"Querying model {self.get_id()} with new prompt.")
            # Query the model.
            completions = self._sample_completions(
                prompt, imgs, temperature, seed, num_completions
            )
            # Cache the completion.
            cache_str = prompt + _CACHE_SEP + _CACHE_SEP.join(completions)
            with open(cache_filepath, "w", encoding="utf-8") as f:
                f.write(cache_str)
            if imgs is not None:
                # Also save the images for easy debugging.
                imgs_folderpath = cache_filepath.parent / "imgs"
                os.makedirs(imgs_folderpath, exist_ok=True)
                for i, img in enumerate(imgs):
                    filename_suffix = str(i) + ".jpg"
                    img.save(os.path.join(imgs_folderpath, filename_suffix))
            logging.debug(f"Saved model response to {cache_filepath}.")
        # Load the saved completion.
        with open(cache_filepath, "r", encoding="utf-8") as f:
            cache_str = f.read()
        logging.debug(f"Loaded model response from {cache_filepath}.")
        assert cache_str.count(_CACHE_SEP) == num_completions
        cached_prompt, completion_strs = cache_str.split(_CACHE_SEP, 1)
        assert cached_prompt == prompt
        completions = completion_strs.split(_CACHE_SEP)
        return completions

    @abc.abstractmethod
    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        """Return log probabilities for a multiple choice question."""

    def _get_cache_filepath(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
        filename: str = "prompt.txt",
    ) -> Path:

        # Set up the cache file.
        assert _CACHE_SEP not in prompt
        os.makedirs(self._cache_dir, exist_ok=True)
        model_id = self.get_id()
        prompt_id = consistent_hash(prompt)
        config_id = f"{temperature}_{seed}_{num_completions}"
        # If the temperature is 0, the seed does not matter.
        if temperature == 0.0:
            config_id = f"most_likely_{num_completions}"
        cache_foldername = f"{model_id}_{config_id}_{prompt_id}"
        if imgs is not None:
            # We also need to hash all the images in the prompt.
            img_hash_list: list[str] = []
            for img in imgs:
                img_hash_list.append(str(imagehash.phash(img)))
            # NOTE: it's possible that this string (the concatenated hashes of
            # each image) is very long. This would make the final cache
            # foldername long. In many operating systems, the maximum folder
            # name length is 255 characters. To shorten this foldername more, we
            # can hash this string into a shorter string. For example, look at
            # https://stackoverflow.com/questions/57263436/hash-like-string-shortener-with-decoder  # pylint:disable=line-too-long
            imgs_id = consistent_hash("".join(img_hash_list))
            cache_foldername += f"{imgs_id}"
        cache_folderpath = self._cache_dir / cache_foldername
        os.makedirs(cache_folderpath, exist_ok=True)
        return cache_folderpath / filename


class VisionLanguageModel(PretrainedLargeModel):
    """A class for all VLM's."""

    def sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        assert imgs is not None
        return super().sample_completions(
            prompt, imgs, temperature, seed, num_completions
        )


class LargeLanguageModel(PretrainedLargeModel):
    """A class for all LLM's."""

    def sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        assert imgs is None
        return super().sample_completions(
            prompt, imgs, temperature, seed, num_completions
        )


class OpenAIModel:
    """Common interface with methods for all OpenAI-based models."""

    def __init__(self, model_name: str, max_tokens: int = 700) -> None:
        """See https://platform.openai.com/docs/models for the list of
        available model names."""
        self._model_name = model_name
        # Note that max_tokens is the maximum response length (not prompt).
        # From OpenAI docs: "The token count of your prompt plus max_tokens
        # cannot exceed the model's context length."
        self._max_tokens = max_tokens
        self.set_openai_key()

    def set_openai_key(self, key: str | None = None) -> None:
        """Set the OpenAI API key."""
        if key is None:
            assert "OPENAI_API_KEY" in os.environ
            key = os.environ["OPENAI_API_KEY"]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def call_openai_api(
        self,
        messages: list,
        model: str = "gpt-4",
        seed: int | None = None,
        max_tokens: int = 32,
        temperature: float = 0.2,
        verbose: bool = False,
    ) -> str:
        """Make an API call to OpenAI."""
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        if verbose:
            logging.debug(f"OpenAI API response: {completion}")
        assert len(completion.choices) == 1
        assert completion.choices[0].message.content is not None
        return completion.choices[0].message.content


class GoogleGeminiModel:
    """Common interface and methods for all Gemini-based models.

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    def __init__(self, model_name: str) -> None:
        """See https://ai.google.dev/models/gemini for the list of available
        model names."""
        self._model_name = model_name
        assert "GOOGLE_API_KEY" in os.environ
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = genai.GenerativeModel(self._model_name)


class OpenAILLM(LargeLanguageModel, OpenAIModel):
    """Interface to openAI LLMs.

    Assumes that an environment variable OPENAI_API_KEY is set to a
    private API key for beta.openai.com.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        max_tokens: int = 700,
        use_cache_only: bool = False,
    ) -> None:
        LargeLanguageModel.__init__(self, cache_dir, use_cache_only)
        OpenAIModel.__init__(self, model_name, max_tokens)

    def get_id(self) -> str:
        return f"openai-{self._model_name}"

    def _sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        del imgs  # unused
        messages = [{"role": "user", "content": prompt, "type": "text"}]
        responses = [
            self.call_openai_api(
                messages,
                model=self._model_name,
                seed=seed,
                max_tokens=self._max_tokens,
                temperature=temperature,
            )
            for _ in range(num_completions)
        ]
        return responses

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        """Return log probabilities for a multiple choice question."""
        choices_prompt = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        extended_prompt = f"""{prompt}

Given the choices below, return the number corresponding to your answer. Do not say anything except the number.

Choices:
{choices_prompt}
"""
        # Handle caching: use JSON format for this.
        cache_filepath = self._get_cache_filepath(
            prompt=extended_prompt,
            imgs=None,
            temperature=0.0,
            seed=seed,
            filename="multiple_choice.json",
        )

        if not cache_filepath.exists():
            logging.debug(f"Querying model {self.get_id()} with new prompt.")
            messages = [{"role": "user", "content": extended_prompt, "type": "text"}]
            client = openai.OpenAI()
            completion = client.chat.completions.create(
                model=self._model_name,
                messages=messages,  # type: ignore
                seed=seed,
                logprobs=True,
                top_logprobs=len(choices),
            )
            logprobs = completion.choices[0].logprobs
            assert logprobs is not None
            assert logprobs.content is not None
            top_logprobs = logprobs.content[0].top_logprobs
            token_to_logprob = {c.token: c.logprob for c in top_logprobs}
            choice_to_logprob: dict[str, float] = {}
            for i, choice in enumerate(choices):
                logprob = token_to_logprob.get(f"{i+1}", -float("inf"))
                choice_to_logprob[choice] = logprob
            with open(cache_filepath, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "prompt": prompt,
                        "choices": choices,
                        "logprobs": choice_to_logprob,
                    },
                    fp,
                )
            logging.debug(f"Saved model response to {cache_filepath}.")
        with open(cache_filepath, "r", encoding="utf-8") as fp:
            choice_to_logprob = json.load(fp)["logprobs"]
        return choice_to_logprob


class GoogleGeminiLLM(LargeLanguageModel, GoogleGeminiModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    def __init__(
        self, model_name: str, cache_dir: Path, use_cache_only: bool = False
    ) -> None:
        LargeLanguageModel.__init__(self, cache_dir, use_cache_only)
        GoogleGeminiModel.__init__(self, model_name)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        del seed  # unused
        assert imgs is None
        generation_config = genai.types.GenerationConfig(  # pylint:disable=no-member
            candidate_count=num_completions, temperature=temperature
        )
        response = self._model.generate_content(
            [prompt], generation_config=generation_config
        )  # type: ignore
        response.resolve()
        return [response.text]

    def get_id(self) -> str:
        return f"Google-{self._model_name}"

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        raise NotImplementedError("TODO")


class GoogleGeminiVLM(VisionLanguageModel, GoogleGeminiModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    def __init__(
        self, model_name: str, cache_dir: Path, use_cache_only: bool = False
    ) -> None:
        VisionLanguageModel.__init__(self, cache_dir, use_cache_only)
        GoogleGeminiModel.__init__(self, model_name)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
    def _sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        del seed  # unused
        assert imgs is not None
        generation_config = genai.types.GenerationConfig(  # pylint:disable=no-member
            candidate_count=num_completions, temperature=temperature
        )
        response = self._model.generate_content(
            [prompt] + imgs, generation_config=generation_config
        )  # type: ignore
        response.resolve()
        return [response.text]

    def get_id(self) -> str:
        return f"Google-{self._model_name}"

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        raise NotImplementedError("TODO")


class OpenAIVLM(VisionLanguageModel, OpenAIModel):
    """Interface for OpenAI's VLMs, including GPT-4 Turbo (and preview
    versions)."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        max_tokens: int = 700,
        use_cache_only: bool = False,
    ) -> None:
        VisionLanguageModel.__init__(self, cache_dir, use_cache_only)
        OpenAIModel.__init__(self, model_name, max_tokens)

    def prepare_vision_messages(
        self,
        images: list[PIL.Image.Image],
        prefix: str | None = None,
        suffix: str | None = None,
        image_size: int = 512,
        detail: str = "auto",
    ) -> list[dict[str, str | list[dict[str, str | Collection[str]]]]]:
        """Prepare text and image messages for the OpenAI API."""
        content: list[dict[str, str | Collection[str]]] = []
        if prefix:
            content.append({"text": prefix, "type": "text"})
        assert images
        assert detail in ["auto", "low", "high"]
        for img in images:
            img_resized = img
            if image_size:
                factor = image_size / max(img.size)
                img_resized = img.resize(
                    (int(img.size[0] * factor), int(img.size[1] * factor))
                )
            # Convert the image to PNG format and encode it in base64
            buffer = BytesIO()
            img_resized.save(buffer, format="PNG")
            buf = buffer.getvalue()
            frame = base64.b64encode(buf).decode("utf-8")
            content_str = {
                "image_url": {
                    "url": f"data:image/png;base64,{frame}",
                    "detail": "auto",
                },
                "type": "image_url",
            }
            content.append(content_str)
        if suffix:
            content.append({"text": suffix, "type": "text"})
        return [{"role": "user", "content": content}]

    def get_id(self) -> str:
        """Get an identifier for the model."""
        return f"OpenAI-{self._model_name}"

    def _sample_completions(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None,
        temperature: float,
        seed: int,
        num_completions: int = 1,
    ) -> list[str]:
        """Query the model and get responses."""
        del seed  # unused.
        if imgs is None:
            raise ValueError("images cannot be None")
        messages = self.prepare_vision_messages(
            prefix=prompt, images=imgs, detail="auto"
        )
        responses = [
            self.call_openai_api(
                messages,
                model=self._model_name,
                max_tokens=self._max_tokens,
                temperature=temperature,
            )
            for _ in range(num_completions)
        ]
        return responses

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        raise NotImplementedError("TODO")


@dataclass(frozen=True)
class SynthesizedPythonFunction:
    """Wrapper around a Python function."""

    function_name: str
    code_str: str

    @cached_property
    def filepath(self) -> Path:
        """Get a file with the code string implemented in it."""
        filename = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".py").name)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.code_str)
        return filename

    def run(self, input_args: list[Any]) -> Any:
        """Run the function on an input (that will be unpacked)."""
        # Import get_plan().
        module_name = f"{self.filepath.stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.filepath)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        assert module is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        # Run the generalized plan.
        return getattr(module, self.function_name)(*input_args)  # type: ignore  # pylint: disable=undefined-variable


_PREVIOUS_SYNTHESIZED_PROMPT = string.Template(
    """
Previously, you synthesized this function:

$function
"""
)


class _PythonFunctionValidationResult:
    """Either a success or failure."""

    @abc.abstractmethod
    def get_prompt_addendum(self) -> str:
        """Get a prompt to correct the program."""


class _PythonFunctionValidationSuccess(_PythonFunctionValidationResult):
    """Indicates all checks passed."""

    def get_prompt_addendum(self) -> str:
        raise NotImplementedError


class _PythonFunctionValidationFailure(_PythonFunctionValidationResult):
    """Some check failed."""


@dataclass(frozen=True)
class _ExceptionPythonFunctionValidationFailure(_PythonFunctionValidationFailure):
    """An exception was raised while trying to execute the python function."""

    input_args: list[Any]
    traceback_str: str

    def get_prompt_addendum(self) -> str:
        return f"""Given these input arguments:

{self.get_input_arg_str()}
        
the following exception was raised:
        
{self.traceback_str}

Fix the code.
"""

    def get_input_arg_str(self) -> str:
        """Get a string representation of the input arguments."""
        return ", ".join([str(x) for x in self.input_args])


@dataclass(frozen=True)
class _TimeOutPythonFunctionValidationFailure(_PythonFunctionValidationFailure):
    """The code ran out of time while executing."""

    def get_prompt_addendum(self) -> str:
        return f"""The code timed out while executing. There may be an infinite loop.

Fix the code.
"""


@dataclass(frozen=True)
class _ExpectedOutputPythonFunctionValidationFailure(_PythonFunctionValidationFailure):
    """The code produced an incorrect output."""

    input_args: list[Any]
    expected_output: Any
    output: Any

    def get_prompt_addendum(self) -> str:
        return f"""Given these input arguments:

{self.get_input_arg_str()}
        
the code produced this output:

{self.output}

but the following output was expected:

{self.expected_output}

Fix the code.
"""

    def get_input_arg_str(self) -> str:
        """Get a string representation of the input arguments."""
        return ", ".join([str(x) for x in self.input_args])


@dataclass(frozen=True)
class SynthesisInfo:
    """Additional info returned along with a synthesized program."""

    result: _PythonFunctionValidationResult
    optimized_args: dict[int, Any]  # arg index to value

    @property
    def success(self) -> bool:
        """Whether the synthesis was a success."""
        return isinstance(self.result, _PythonFunctionValidationSuccess)


class SynthesizedProgramArgumentOptimizer:
    """Optimizes certain arguments of a synthesized function."""

    def __init__(self, arg_idx_to_space: dict[int, Space]):
        self._arg_idx_to_space = arg_idx_to_space

    @abc.abstractmethod
    def optimize(
        self,
        synthesized_function: SynthesizedPythonFunction,
        input_output_examples: list[tuple[list[Any], Any]],
        outputs_equal_check: Callable[[Any, Any], bool] = operator.eq,
    ) -> dict[int, Any]:
        """Run optimization."""

    def get_optimized_args_from_input(self, input_args: list[Any]) -> dict[int, Any]:
        """Extract optimized arguments from an input."""
        return {idx: input_args[idx] for idx in self._arg_idx_to_space}

    def substitute_optimized_args(
        self, input_args: list[Any], optimized_args: dict[int, Any]
    ) -> list[Any]:
        """Run substitution."""
        new_args = list(input_args)
        for idx, value in optimized_args.items():
            new_args[idx] = value
        return new_args


class NullSynthesizedProgramArgumentOptimizer(SynthesizedProgramArgumentOptimizer):
    """Doesn't actually run any optimization."""

    def optimize(
        self,
        synthesized_function: SynthesizedPythonFunction,
        input_output_examples: list[tuple[list[Any], Any]],
        outputs_equal_check: Callable[[Any, Any], bool] = operator.eq,
    ) -> dict[int, Any]:
        return self.get_optimized_args_from_input(input_output_examples[0][0])


class GridSearchSynthesizedProgramArgumentOptimizer(
    SynthesizedProgramArgumentOptimizer
):
    """Optimizes by running a grid search over Box spaces."""

    def __init__(self, arg_idx_to_space: dict[int, Space], num_grid_steps: int = 10):
        super().__init__(arg_idx_to_space)
        assert all(isinstance(space, Box) for space in arg_idx_to_space.values())

    def optimize(
        self,
        synthesized_function: SynthesizedPythonFunction,
        input_output_examples: list[tuple[list[Any], Any]],
        outputs_equal_check: Callable[[Any, Any], bool] = operator.eq,
    ) -> dict[int, Any]:
        return {1: 0.05, 2: 0.95}  # TODO


def synthesize_python_function_with_llm(
    llm: LargeLanguageModel,
    function_name: str,
    input_output_examples: list[tuple[list[Any], Any]],
    initial_prompt: str,
    code_prefix: str = "",
    previous_synthesized_prompt: string.Template = _PREVIOUS_SYNTHESIZED_PROMPT,
    outputs_equal_check: Callable[[Any, Any], bool] = operator.eq,
    argument_optimizer: SynthesizedProgramArgumentOptimizer | None = None,
    timeout: int = 30,
    max_debug_attempts: int = 5,
    temperature: float = 0.0,
    seed: int = 0,
) -> tuple[SynthesizedPythonFunction, SynthesisInfo]:
    """Use an LLM to synthesize a python function.

    The input output examples are each of the form (args, output).
    """
    if argument_optimizer is None:
        argument_optimizer = NullSynthesizedProgramArgumentOptimizer({})
    prompt = initial_prompt
    synthesized_function: SynthesizedPythonFunction | None = None
    synthesis_info: SynthesisInfo | None = None
    for _ in range(max_debug_attempts):
        response = llm.sample_completions(
            prompt, imgs=None, temperature=temperature, seed=seed
        )[0]
        python_function_str = parse_python_code_from_llm_response(response)
        python_function_str = code_prefix + python_function_str
        synthesized_function = SynthesizedPythonFunction(
            function_name, python_function_str
        )
        synthesis_info = optimize_llm_generated_python_function(
            synthesized_function,
            input_output_examples,
            argument_optimizer=argument_optimizer,
            outputs_equal_check=outputs_equal_check,
            timeout=timeout,
        )
        # Success!
        if synthesis_info.success:
            return synthesized_function, synthesis_info
        # Failure, reprompt.
        previous_prompt = previous_synthesized_prompt.substitute(function=response)
        error_prompt = synthesis_info.result.get_prompt_addendum()
        prompt = initial_prompt + previous_prompt + error_prompt
    # If all debug attempts are exceeded, return the last program and report failure.
    assert synthesized_function is not None
    assert synthesis_info is not None
    return synthesized_function, synthesis_info


def optimize_llm_generated_python_function(
    synthesized_function: SynthesizedPythonFunction,
    input_output_examples: list[tuple[list[Any], Any]],
    argument_optimizer: SynthesizedProgramArgumentOptimizer,
    outputs_equal_check: Callable[[Any, Any], bool] = operator.eq,
    timeout: int = 30,
) -> SynthesisInfo:
    """Check for execution errors, timeouts, and input-output failures."""
    # Handle possible timeouts.
    manager = mp.Manager()
    result_proxy_dict = manager.dict()
    p = mp.Process(
        target=_optimize_llm_generated_python_function_no_timeout,
        args=(
            synthesized_function,
            input_output_examples,
            argument_optimizer,
            outputs_equal_check,
            result_proxy_dict,
        ),
    )
    p.start()
    p.join(timeout)
    result_dict = dict(result_proxy_dict)
    # Timeout reached.
    if p.is_alive():
        # Treated like a KeyboardInterrupt.
        assert p.pid is not None
        os.kill(p.pid, signal.SIGINT)
        # Give it a few more seconds then kill for good.
        p.join(3)
        p.kill()
        # Return timeout.
        validation_result: _PythonFunctionValidationResult = (
            _TimeOutPythonFunctionValidationFailure()
        )
    else:
        # Did not time out.
        validation_result = result_dict["validation_result"]
    # Finish synthesis info.
    return SynthesisInfo(validation_result, result_dict["optimized_args"])


def _optimize_llm_generated_python_function_no_timeout(
    synthesized_function: SynthesizedPythonFunction,
    input_output_examples: list[tuple[list[Any], Any]],
    argument_optimizer: SynthesizedProgramArgumentOptimizer,
    outputs_equal_check: Callable[[Any, Any], bool],
    result_dict: dict,
) -> None:

    # Initialize the optimization arguments using the first input-output example.
    result_dict["optimized_args"] = argument_optimizer.get_optimized_args_from_input(
        input_output_examples[0][0]
    )

    # Run argument optimization.
    try:
        optimized_args = argument_optimizer.optimize(
            synthesized_function, input_output_examples, outputs_equal_check
        )
        result_dict["optimized_args"] = optimized_args
    except BaseException as e:  # pylint: disable=broad-exception-caught
        pass  # raise the failure below instead because we don't know input idx

    # This might be redundant; look into refactoring.
    optimize_args = result_dict["optimized_args"]
    for input_args, expected_output in input_output_examples:
        input_args = argument_optimizer.substitute_optimized_args(
            input_args, optimize_args
        )
        try:
            output = synthesized_function.run(input_args)
        except BaseException as e:  # pylint: disable=broad-exception-caught
            result_dict["validation_result"] = _get_validation_failure_from_exception(
                e, input_args, synthesized_function
            )
            return
        # Output failure.
        if not outputs_equal_check(output, expected_output):
            result_dict["validation_result"] = (
                _ExpectedOutputPythonFunctionValidationFailure(
                    input_args, expected_output, output
                )
            )
            return
    # All successful!
    result_dict["validation_result"] = _PythonFunctionValidationSuccess()
    return


def parse_python_code_from_llm_response(response: str) -> str:
    """Parse Python code from an LLM response, assuming ```python tag."""
    # Parse out python code if it exists.
    python_code_prefix = "```python"
    if python_code_prefix in response:
        python_start = response.index(python_code_prefix)
        python_remainder = response[python_start + len(python_code_prefix) :]
        if "```" in python_remainder:
            python_end = python_remainder.index("```")
        else:
            python_end = len(python_remainder)
        python_response = python_remainder[:python_end]
        return python_response
    return response


def _get_validation_failure_from_exception(
    e: BaseException,
    input_args: list[Any],
    synthesized_function: SynthesizedPythonFunction,
) -> _ExceptionPythonFunctionValidationFailure:
    tb = traceback.format_exception(e)
    tb_lines = [
        l.replace(str(synthesized_function.filepath), "<file-name-omitted>")
        for l in tb
        if "tomsutils" not in l
    ]
    tb_str = "".join(tb_lines)
    return _ExceptionPythonFunctionValidationFailure(input_args, tb_str)
