"""
This package provides LLM-specific detectors for the automatic model scan.
"""
from .llm_basic_sycophancy_detector import LLMBasicSycophancyDetector
from .llm_chars_injection_detector import LLMCharsInjectionDetector
from .llm_harmful_content_detector import LLMHarmfulContentDetector
from .llm_implausible_output_detector import LLMImplausibleOutputDetector
from .llm_information_disclosure_detector import LLMInformationDisclosureDetector
from .llm_output_formatting_detector import LLMOutputFormattingDetector
from .llm_prompt_injection_detector import LLMPromptInjectionDetector
from .llm_stereotypes_detector import LLMStereotypesDetector
from .llm_numerical_faithfulness_detector import LLMNumericalFaithfulnessDetector

__all__ = [
    "LLMBasicSycophancyDetector",
    "LLMCharsInjectionDetector",
    "LLMHarmfulContentDetector",
    "LLMImplausibleOutputDetector",
    "LLMInformationDisclosureDetector",
    "LLMOutputFormattingDetector",
    "LLMPromptInjectionDetector",
    "LLMStereotypesDetector",
    "LLMNumericalFaithfulnessDetector",
]
