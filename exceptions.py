

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


class LLMApiTimeoutError(LLMExceptions):
    """
    LLM Exception to be raised in case of model raises timeout error
    """
    type = LLMPredictionErrors.LLM_API_TIMEOUT.value


class LLMApiConnectionError(LLMExceptions):
    """
    LLM Exception to be raised in case of model raises api error
    """
    type = LLMPredictionErrors.LLM_API_CONNECTION_ERROR.value


class LLMApiAuthenticationError(LLMExceptions):
    """
    LLM Exception to be raised in case of model raises authentication error
    """
    type = LLMPredictionErrors.LLM_API_AUTHENTICATION_ERROR.value


class LLMApiRateLimitError(LLMExceptions):
    """
    LLM Exception to be raised in case of model raises rate limit error
    """
    type = LLMPredictionErrors.LLM_API_RATE_LIMIT_REACHED.value


class LLMApiBadRequestError(LLMExceptions):
    """
    LLM Exception to be raised in case of model raises bad request error
    """
    type = LLMPredictionErrors.LLM_API_BAD_REQUEST_ERROR.value


class LLMApiNotFoundError(LLMExceptions):
    """
    LLM Exception to be raised in case of model raises 404 not found
    """
    type = LLMPredictionErrors.LLM_API_NOT_FOUND_ERROR.value


class LLMApiInternalServerError(LLMExceptions):
    """
    LLM Exception to be raised in case of model raises internal server error
    """
    type = LLMPredictionErrors.LLM_API_INTERNAL_SERVER_ERROR.value


# Below exceptions are related to issue happening at our code level
class DSExceptions(CustomException):
    """
    Base exception class to all DS Internal Exceptions during prediction
    """
    type = "DS_EXCEPTION"


class DSInternalServerError(DSExceptions):
    """
    Exception to be raised in case of ds code failure while running prediction
    """
    type = LLMPredictionErrors.DS_INTERNAL_SERVER_ERROR.value


class BadRequestException(DSExceptions):
    """
    Exception to be raised in case of bad request where properties of bot or request couldn't be found
    """
    type = LLMPredictionErrors.BAD_REQUEST_ERROR.value

