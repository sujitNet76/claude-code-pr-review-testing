class CustomException(Exception):
    """
    Base exception class to all LLM Exceptions
    """
    type = "Type of exception"


class LLMExceptions(CustomException):
    """
    Base exception class to all LLM Exceptions
    """
    type = "LLM_EXCEPTION"