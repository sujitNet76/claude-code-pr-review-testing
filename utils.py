import os
import asyncio
from datetime import datetime

from conversation_context_preprocessor.conversation_context_preprocessor import ConversationContextPreprocessor
from fastapi import Request
from langchain_core.messages import messages_from_dict
from common.formatters.chat_history.utils import remove_tool_calls
from common.llm_model.llm_config_data import LLMConfigData
from common.utils.llm_utils import add_credential_in_llm_tasks_config
from connections.async_kafka.aiokafka_producer import AsyncKafkaProducerSingleton
from connections.kafka.kafka_producer import KafkaProducer
from configurations.config import get_config, get_dm_config_proxy, get_openai_tracker
import openai
from modules.llm_prediction_service.exceptions import (
    CustomException,
    LLMExceptions,
    LLMApiTimeoutError,
    LLMApiConnectionError,
    LLMApiAuthenticationError,
    LLMApiRateLimitError,
    LLMApiBadRequestError,
    LLMApiNotFoundError,
    LLMApiInternalServerError,
)
from common.llm_model.constants import DefaultResponseGenerationModel, LLMServiceType, DefaultAgenticOSModel
from apps.llm_prediction.models import LLMAnsRequest, LLMAnsResponse, LLMAnsSyncRequest, RetrieverRequest
from common.models.conversation_tracker_event import ConversationTrackerEventObject
from modules.config_manager.config_manager import ConfigManager
from common.constants import Envs, ResponseStreamErrorMessage
from loguru import logger
from common.utils.utils import request_timing_context
from typing import Optional


configs = get_config()
dm_config_proxy = get_dm_config_proxy()
openai_status_tracker = get_openai_tracker()
kafka_utils = KafkaProducer()
async_kafka_utils = AsyncKafkaProducerSingleton()
config_manager = ConfigManager()




def produce_error_in_kafka(exception: CustomException, request_id: str, conversation_id: str,
                           bot_id: str, bot_ref_id: str):
    error_message = ResponseStreamErrorMessage(
        triggerType="EVENT",
        eventType="ERROR",
        subType=exception.type,
        requestId=request_id,
        conversationId=conversation_id, 
        botId=bot_id,
        botRefId=bot_ref_id
    )
    kafka_utils.produce_message(topic=configs.ORCHESTRATOR_ERROR_KAFKA_TOPIC_NAME, key=conversation_id,
                                value=error_message.to_json(), log_message=True)
    kafka_utils.poll(timeout_seconds=0)


def get_openai_llm_exception(e):
    error_map = {
        openai.APITimeoutError: LLMApiTimeoutError,
        openai.APIConnectionError: LLMApiConnectionError,
        openai.AuthenticationError: LLMApiAuthenticationError,
        openai.RateLimitError: LLMApiRateLimitError,
        openai.BadRequestError: LLMApiBadRequestError,
        openai.NotFoundError: LLMApiNotFoundError,
        openai.InternalServerError: LLMApiInternalServerError,
    }
    custom_error = error_map.get(type(e), LLMExceptions)
    return custom_error(e)


def get_llm_info_by_service_name(llm_infos: list[dict], service_name: str) -> LLMConfigData:
    for llm_info in llm_infos:
        config_dict = llm_info.get("selectedGenerativeAIAlgorithm", {})
        if (llm_info.get("serviceName", "") == service_name) or (config_dict.get("serviceName", "") == service_name):
            task_config = config_dict.get('taskConfig')
            if not task_config:
                # NOTE: This condition is for temporary release
                if service_name == LLMServiceType.MODEL_CREATION.value:
                    task_config = [{'task': 'EMBEDDING_CREATION',
                                    'provider': llm_info['selectedGenerativeAIAlgorithm']['generativeAIModels'][0]['serviceProvider'],
                                    'task_type': 'TEXT_EMBEDDING',
                                    'model_name': llm_info['selectedGenerativeAIAlgorithm']['generativeAIModels'][0]['modelName'],
                                    'model_type': llm_info['selectedGenerativeAIAlgorithm']['generativeAIModels'][0]['modelType'],
                                    'properties': {'LLM_RETRIEVER_THRESHOLD_CONFIG': 0.75}}]
                else:
                    return None

            model = config_dict["generativeAIModels"][0]

            integration = config_dict["genAiIntegration"]
            credentials = None
            if integration:
                credentials = integration.get("credentials", None)
            return LLMConfigData(service_name=service_name,
                                 algorithm=config_dict['algorithmIdentifier'],
                                 provider=model['serviceProvider'],
                                 credentials=credentials,
                                 task_config=task_config
                                 )
    return None


def get_default_llm_info(service_name):
    """
    Retrieves default LLM configuration based on service type.

    Args:
        service_name: Type of service ('RESPONSE_GENERATION' or 'AGENTIC_OS').

    Returns:
        LLMConfigData: Configuration containing service name, algorithm, provider, credentials and task config.
        None: If service_name is invalid.
    """
    if service_name == LLMServiceType.RESPONSE_GENERATION.value:
        model_id = DefaultResponseGenerationModel.NETOMI_RESPONSE_GENERATION_MODEL_V7.value
        task_config = configs.DEFAULT_RESPONSE_GENERATION_MODEL_CONFIG
    elif service_name == LLMServiceType.AGENTIC_OS.value:
        model_id = DefaultAgenticOSModel.NETOMI_AGENTIC_OS_MODEL_V1.value
        task_config = configs.DEFAULT_AGENT_LLM_MODEL_CONFIG
    else:
        return None

    provider = task_config[0]['provider']
    return LLMConfigData(service_name=service_name,
                         algorithm=model_id,
                         provider=provider,
                         credentials=None,
                         task_config=task_config)


async def get_llm_info(request: LLMAnsSyncRequest | RetrieverRequest | LLMAnsRequest, service_name: str,
                       *, override_use_cache: Optional[bool] = None) -> LLMConfigData:

    """
    Retrieves LLM configuration from API or falls back to default configuration.

    Args:
        request: The request containing bot ID, LLM name and environment.
        service_name: Type of service ('RESPONSE_GENERATION' or 'AGENTIC_OS').

    Returns:
        LLMConfigData: Configuration with service name, algorithm, provider, credentials and task config.
        None: If service_name is invalid or configuration retrieval fails.
    """

    llm_name = request.llm_name
    bot_id = request.bot_id
    mode = request.env
    # Default behavior
    default_use_cache = str(mode).upper() == Envs.LIVE.value

    # Allow explicit override for special callers, in this case, Prompt Analyzer
    use_cache = default_use_cache if override_use_cache is None else override_use_cache
    
    use_fallback = False
    if not configs.AZURE_APIM_ENABLED:
        use_fallback = not await openai_status_tracker.is_available_async()

    if service_name == LLMServiceType.MODEL_CREATION.value:
        llm_infos = await dm_config_proxy.fetch_llm_info_from_api(bot_id=bot_id, model_name=llm_name,
                                                                  use_fallback=use_fallback, use_cache=use_cache)
        llm_infos = llm_infos.get('llmMappedModelInfoWithServices', None)
        if llm_infos:
            llm_info = get_llm_info_by_service_name(llm_infos=llm_infos, service_name=service_name)
            if llm_info:
                llm_info = add_credential_in_llm_tasks_config(llm_info=llm_info, bot_id=bot_id)
            else:
                logger.warning(f"Using default configuration for bot_id: {bot_id}")
                return get_default_llm_info(service_name=service_name)
        else:
            logger.warning(f"Didn't get data from api using default configuration for bot_id: {bot_id}")
            return get_default_llm_info(service_name=service_name)



    elif service_name in [LLMServiceType.AGENTIC_OS.value, LLMServiceType.RESPONSE_GENERATION.value, LLMServiceType.AGENT_ASSIST_COPILOT.value]:
        llm_infos, gai_priority_access_config = await asyncio.gather(
                    dm_config_proxy.get_llm_config(bot_id=bot_id, service_name=service_name, env=str(request.env).upper(),
                                                    integration_credentials=True, use_fallback=use_fallback, use_cache=use_cache),
                    dm_config_proxy.get_gai_priority_access_config(bot_id=bot_id, env=mode)
        )
        is_llm_priority_tier_enabled = (gai_priority_access_config.get("generativeAIConfigurations", {}).get("isPriorityAccessEnabled", True))
        if llm_infos:
            llm_info = get_llm_info_by_service_name(llm_infos=llm_infos, service_name=service_name)
            if llm_info:
                llm_info = add_credential_in_llm_tasks_config(llm_info, bot_id=bot_id, is_llm_priority_tier_enabled=is_llm_priority_tier_enabled)
            else:
                logger.warning(f"Using default configuration for bot_id: {bot_id}")
                return get_default_llm_info(service_name=service_name)
        else:
            logger.warning(f"Didn't get data from agent api using default configuration for bot_id: {bot_id}")
            return get_default_llm_info(service_name=service_name)

    else:
        logger.error(f"Wrong llm info type provided. Using default response generation configuration.")
        return None

    return llm_info

async def produce_llm_ans_event_in_conversation_tracker(state: str, request_obj: Request,
                                                        request: LLMAnsRequest | LLMAnsResponse,
                                                        use_async_kafka=False, send_metrics=False):
    request_id = request.request_id
    bot_id = request.bot_id
    bot_ref_id = request.bot_ref_id
    conversation_id = request.conversation_id

    _timestamp = datetime.utcnow().isoformat(sep="T", timespec='milliseconds') + 'Z'
    request_json = request.model_dump(by_alias=True).__str__()
    llm_metrics = dict()
    if send_metrics:
        llm_metrics = request_timing_context.get()
    topic = configs.CONVERSATION_EVENT_TRACKER_KAFKA_TOPIC_NAME
    event_tracker_object = ConversationTrackerEventObject(env=configs.ENV_CONFIG, request_json=request_json,
                                                          bot_id=bot_id, bot_ref_id=bot_ref_id,
                                                          conversation_id=conversation_id, request_id=request_id,
                                                          request_status="PENDING",
                                                          service=os.getenv("SERVICE_NAME", "ds-llm-service"),
                                                          current_state=state,
                                                          timestamp=_timestamp,
                                                          dddate=_timestamp,
                                                          llm_metrics=llm_metrics
                                                          )
    event_tracker_json = event_tracker_object.model_dump()

    if use_async_kafka:
        result = await async_kafka_utils.produce_message(topic=topic, key=conversation_id,
                                                         value=event_tracker_json, log_message=True)
    else:
        result = kafka_utils.produce_message(topic=topic, key=conversation_id,
                                                     value=event_tracker_json, log_message=True)
    kafka_utils.poll(timeout_seconds=0)


def exclude_field_from_user_profile(user_profile: Optional[list]) -> Optional[list]:
    try:
        # Exclude date and time info already present in the system prompt to avoid duplication.
        excluded_fields = {"user_date_time", "user_timezone"}

        if user_profile and isinstance(user_profile, list):
            user_profile = [
                    p for p in user_profile
                    if p['name'] and p['name'] not in excluded_fields
                ]
    except Exception as e:
        logger.exception(f"got error [{e}] while removing time and date from user profile")
    return user_profile


def format_chat_history(chat_history: list, is_rag_chain_enabled: bool, include_tool_call: bool):
    formatted_chat_history = []
    try:
        if not include_tool_call:
            chat_history = remove_tool_calls(chat_history=chat_history)
        if chat_history and is_rag_chain_enabled:
            preprocess_context_history = ConversationContextPreprocessor(remove_pattern_list=["🌟", "🌈", "🚀"],
                                                                         remove_emojis=configs.REMOVE_EMOJIS,
                                                                         remove_urls=configs.REMOVE_URLS, remove_html_tags=configs.REMOVE_HTML_TAGS)
            chat_history = preprocess_context_history.preprocess_context(chat_history)
        formatted_chat_history = messages_from_dict(chat_history)
    except Exception as ex:
        logger.warning(f"got error [{ex}] while formatting chat history with error")
    return formatted_chat_history

  
def update_field_in_user_profile(user_profile, fields_dict : dict):
    updated_fields = set()
    for item in user_profile:
        if item.get("name") in fields_dict:
            item["value"] = fields_dict[item.get("name")]
            updated_fields.add(item.get("name"))

    if len(updated_fields) != len(fields_dict):
        for name, value in fields_dict.items():
            if name not in updated_fields:
                user_profile.append({"name": name, "value": value})


