"""Tests for the large language model interface."""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tomsutils.llm import (
    GoogleGeminiLLM,
    GoogleGeminiVLM,
    GridSearchSynthesizedProgramArgumentOptimizer,
    LargeLanguageModel,
    OpenAILLM,
    OpenAIVLM,
    VisionLanguageModel,
    synthesize_python_function_with_llm,
)


class _DummyLLM(LargeLanguageModel):

    def get_id(self) -> str:
        return "dummy"

    def _sample_completions(self, prompt, imgs, temperature, seed, num_completions=1):
        del imgs  # unused.
        completions = []
        for _ in range(num_completions):
            completion = f"Prompt: {prompt}. Seed: {seed}. " f"Temp: {temperature:.1f}."
            completions.append(completion)
        return completions

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        raise NotImplementedError("TODO")


class _MockLLM(LargeLanguageModel):

    def __init__(
        self,
        completions: list[list[str]],
        cache_dir: Path,
        use_cache_only: bool = False,
    ) -> None:
        super().__init__(cache_dir, use_cache_only)
        self.completions = completions

    def get_id(self) -> str:
        return "mock"

    def _sample_completions(self, prompt, imgs, temperature, seed, num_completions=1):
        del imgs  # unused.
        next_completions = self.completions.pop(0)
        assert num_completions == len(next_completions)
        return list(next_completions)

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        raise NotImplementedError("TODO")


class _DummyVLM(VisionLanguageModel):

    def get_id(self) -> str:
        return "dummy"

    def _sample_completions(self, prompt, imgs, temperature, seed, num_completions=1):
        del imgs  # unused.
        completions = []
        for _ in range(num_completions):
            completion = f"Prompt: {prompt}. Seed: {seed}. " f"Temp: {temperature:.1f}."
            completions.append(completion)
        return completions

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> dict[str, float]:
        raise NotImplementedError


def test_large_language_model():
    """Tests for LargeLanguageModel()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    # Query a dummy LLM.
    llm = _DummyLLM(Path(cache_dir.name))
    assert llm.get_id() == "dummy"
    completions = llm.sample_completions("Hello!", None, 0.5, 123, num_completions=3)
    expected_completion = "Prompt: Hello!. Seed: 123. Temp: 0.5."
    assert completions == [expected_completion] * 3
    # Query it again, covering the case where we load from disk.
    completions = llm.sample_completions("Hello!", None, 0.5, 123, num_completions=3)
    assert completions == [expected_completion] * 3
    # Query with temperature 0.
    completions = llm.sample_completions("Hello!", None, 0.0, 123, num_completions=3)
    expected_completion = "Prompt: Hello!. Seed: 123. Temp: 0.0."
    assert completions == [expected_completion] * 3
    # Clean up the cache dir.
    cache_dir.cleanup()
    # Test llm_use_cache_only.
    llm = _DummyLLM(Path(cache_dir.name), use_cache_only=True)
    with pytest.raises(ValueError) as e:
        completions = llm.sample_completions(
            "Hello!", None, 0.5, 123, num_completions=3
        )
    assert "No cached response found for prompt." in str(e)


def test_vision_language_model():
    """Tests for LargeLanguageModel()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    # Query a dummy LLM.
    vlm = _DummyVLM(Path(cache_dir.name))
    assert vlm.get_id() == "dummy"
    dummy_img = Image.new("RGB", (100, 100))
    completions = vlm.sample_completions(
        "Hello!", [dummy_img], 0.5, 123, num_completions=1
    )
    expected_completion = "Prompt: Hello!. Seed: 123. Temp: 0.5."
    assert completions == [expected_completion] * 1
    # Clean up the cache dir.
    cache_dir.cleanup()


def test_openai_llm():
    """Tests for OpenAILLM()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "dummy API key"
    # Create an OpenAILLM.
    llm = OpenAILLM("gpt-4o-mini", Path(cache_dir.name))
    assert llm.get_id() == "openai-gpt-4o-mini"
    # Uncomment this to test manually, but do NOT uncomment in master, because
    # each query costs money.
    # completions = llm.sample_completions("Hi",
    #                                      None,
    #                                      0.5,
    #                                      123,
    #                                      num_completions=2)
    # assert len(completions) == 2
    # completions2 = llm.sample_completions("Hi",
    #                                       None,
    #                                       0.5,
    #                                       123,
    #                                       num_completions=2)
    # assert completions == completions2
    # logprobs = llm.get_multiple_choice_logprobs(
    #     "Fill in the blank with the appropriate homophone: The ____ of the king lasted for ten years.",  # pylint: disable=line-too-long
    #     choices=["rein", "reign", "rain"],
    #     seed=0,
    # )
    # assert logprobs["reign"] > logprobs["rein"]
    # assert logprobs["reign"] > logprobs["rain"]


def test_gemini_vlm():
    """Tests for GoogleGeminiVLM()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "dummy API key"
    # Create an OpenAILLM with the curie model.
    vlm = GoogleGeminiVLM("gemini-pro-vision", Path(cache_dir.name))
    assert vlm.get_id() == "Google-gemini-pro-vision"


def test_gemini_llm():
    """Tests for GoogleGeminiLLM()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "dummy API key"
    # Create an OpenAILLM with the curie model.
    vlm = GoogleGeminiLLM("gemini-1.5-pro", Path(cache_dir.name))
    assert vlm.get_id() == "Google-gemini-1.5-pro"


def test_openai_vlm():
    """Tests for GoogleGeminiVLM()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "dummy API key"
    # Create an OpenAILLM with the curie model.
    vlm = OpenAIVLM("gpt-4-turbo", Path(cache_dir.name))
    assert vlm.get_id() == "OpenAI-gpt-4-turbo"
    dummy_img = Image.new("RGB", (100, 100))
    vision_messages = vlm.prepare_vision_messages([dummy_img], "wakanda", "forever")
    assert len(vision_messages) == 1
    assert vision_messages[0]["content"][1]["type"] == "image_url"  # type: ignore
    # NOTE: Uncomment below lines for actual test.
    # test_img_filepath = Path(__file__).parent / "test_vlm_predicate_img.jpg"
    # images = [Image.open(test_img_filepath)]
    # prompt = """
    #     Describe the object relationships between the objects and containers.
    #     You can use following predicate-style descriptions:
    #     Inside(object1, container)
    #     Blocking(object1, object2)
    #     On(object, surface)
    #     """
    # completions = vlm.sample_completions(prompt=prompt,
    #                                      imgs=images,
    #                                      temperature=0.5,
    #                                      num_completions=3,
    #                                      seed=0)
    # assert len(completions) == 3
    # for completion in completions:
    #     assert "Inside" in completion


def test_synthesize_python_function_with_llm():
    """Tests for synthesize_python_function_with_llm()."""
    function_name = "count_good_dogs"
    prompt = """Generate a Python function of the form
    
def count_good_dogs(dog_names: list[str]) -> int:
    # your code here
"""

    input_output_examples = [([["nomsy", "rover"]], 2), ([["nomsy"]], 1)]

    response_with_execution_error = """```python
def count_good_dogs(dog_names: list[str]) -> int:
    tenth_dog = dog_names[10]
    return 10
```    
"""

    response_with_infinite_loop = """```python
def count_good_dogs(dog_names: list[str]) -> int:
    num_good_dogs = 0
    while True:
        num_good_dogs += 1
    return num_good_dogs
```    
"""

    response_with_semantic_failure = """```python
def count_good_dogs(dog_names: list[str]) -> int:
    return 2
```    
"""

    response_with_correct_answer = """```python
def count_good_dogs(dog_names: list[str]) -> int:
    return len(dog_names)
```    
"""

    completions = [
        [response_with_execution_error],
        [response_with_infinite_loop],
        [response_with_semantic_failure],
        [response_with_correct_answer],
    ]
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    llm = _MockLLM(completions, Path(cache_dir.name))

    fn, success = synthesize_python_function_with_llm(
        llm, function_name, input_output_examples, prompt, timeout=2
    )
    assert success

    for input_args, expected_output in input_output_examples:
        assert fn.run(input_args) == expected_output


def test_synthesize_python_function_with_llm_optimize_arguments():
    """Tests synthesize_python_function_with_llm() with argument
    optimization."""
    function_name = "find_my_thresholds"
    arg_index_to_space_var_name = "ARG_INDEX_TO_SPACE"
    prompt = """Generate a Python function of the form
    
def find_my_thresholds(dog_name: str, threshold1: float, threshold2: float) -> bool:
    # your code here
"""

    input_output_examples: list[tuple[list[Any], Any]] = [
        (["nomsy", 0.5, 0.5], True),
        (["rover", 0.5, 0.5], True),
    ]

    arg_optimizer = GridSearchSynthesizedProgramArgumentOptimizer()
    code_prefix = """from gymnasium.spaces import Box
"""

    response = """```python

ARG_INDEX_TO_SPACE = {
    1: Box(0.0, 1.0, shape=tuple()),  # threshold1
    2: Box(0.0, 1.0, shape=tuple()),  # threshold2
}

def find_my_thresholds(dog_name: str, threshold1: float, threshold2: float) -> bool:
    if dog_name == "nomsy":
        return threshold1 < 0.1
    if dog_name == "rover":
        return threshold2 > 0.9
    raise NotImplementedError
```    
"""

    completions = [
        [response],
        [response],
    ]
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    llm = _MockLLM(completions, Path(cache_dir.name))

    fn, info = synthesize_python_function_with_llm(
        llm,
        function_name,
        input_output_examples,
        prompt,
        timeout=2,
        code_prefix=code_prefix,
        argument_optimizer=arg_optimizer,
        arg_index_to_space_var_name=arg_index_to_space_var_name,
    )
    assert info.success
    assert info.optimized_args[1] < 0.1
    assert info.optimized_args[2] > 0.9

    for input_args, expected_output in input_output_examples:
        input_args = arg_optimizer.substitute_optimized_args(
            input_args, info.optimized_args
        )
        assert fn.run(input_args) == expected_output
