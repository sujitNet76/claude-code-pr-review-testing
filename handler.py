# fmt:off
import asyncio
from loguru import logger
import time
from apps.llm_prediction.models import (LLMAnsRequest, Token)
from common.constants import ApiRequestSourceTypes, CopilotTargetUser
from common.formatters.Prompts.utils import get_formatted_prompts
from apps.llm_prediction.utils import set_params_for_ans_request, anonymize_request, deanonymize_rag_response, \
    is_rag_chain_enabled, is_llm_va_enabled, get_llm_config_models_by_capability
from common.utils.rephrased_utils import get_llm_query
from configurations.config import get_config
from modules.llm_prediction_service.response_streamer.stream_object import FillerObject
from modules.llm_prediction_service.llm_chains.rag_chain import RagChainResponseGeneration
from common.constants import (Capability, CapabilityTasks, ErrorCodes, AiAgentFramework, PromptBotConfigs,
                              ResponseGenerationFramework, AdvancedSeriesModels)
from common.llm_model.constants import DefaultAgenticOSModel
from common.utils.llm_utils import extend_original_chat_history
from common.utils.utils import request_timing_context, total_unique_runtime
from common.utils.postprocess_llm_response import LLMResponseProcessor, AgentResponseProcessor
from common.utils.time_utils import get_utc_timestamp_in_milliseconds, get_time_difference_in_seconds
from connections.kafka.kafka_producer import KafkaProducer
from common.constants import (APIStatus, ConversationTrackerEvents, VerdictType,
                              ResponseType, StreamingState)
from modules.llm_prediction_service.simple_agent.langgraph_agent.agent import LanggraphAgent
from modules.llm_prediction_service.utils import (produce_error_in_kafka,
                                                  format_chat_history)
from fastapi import Request
from modules.llm_prediction_service.utils import produce_llm_ans_event_in_conversation_tracker
from modules.llm_prediction_service.exceptions import LLMExceptions, DSExceptions, BadRequestException
from fastapi import status
from common.constants import ResponseStreamErrorMessage


config = get_config()

kafka_util = KafkaProducer()

def _stringify_custom_fields(custom_fields: list) -> list:
    """Return a deep-copied list of custom fields with values stringified.
    - dict/list values → JSON string (compact)
    - other scalars → str(value)
    """
    import json as _json
    out = []
    if not isinstance(custom_fields, list):
        return out
    for cf in custom_fields:
        if not isinstance(cf, dict):
            continue
        name = cf.get("name")
        values = cf.get("values", [])
        if name is None or values is None:
            continue
        str_values = []
        for v in values if isinstance(values, list) else [values]:
            if isinstance(v, (dict, list)):
                try:
                    str_values.append(_json.dumps(v, ensure_ascii=False, separators=(",", ":")))
                except Exception:
                    str_values.append(str(v))
            else:
                str_values.append(str(v))
        entry = {"name": name, "values": str_values}
        if "id" in cf:
            entry["id"] = cf["id"]
        out.append(entry)
    return out


async def stream_llm_response_background(request_obj: Request, request: LLMAnsRequest):
    time_in = time.time()
    request.config_flags["isRagChainEnabled"] = await is_rag_chain_enabled(request.bot_id)
    request.config_flags["isLLMVaEnabled"] = is_llm_va_enabled(request.bot_id)
    request.config_flags["isAgentAssistEnabled"] = (
            request_obj.headers.get("x-source", "").lower() == ApiRequestSourceTypes.AGENT_ASSIST.value)

    try:
        logger.info(f"request input after setting required parameters before PII Anonymization : {request}")
        log_id = f"Bot_id:{request.bot_id}, conversation_id:{request.conversation_id} :"

        await produce_llm_ans_event_in_conversation_tracker(
            state=ConversationTrackerEvents.DS_GATEWAY_ANS_API_RECEIVED_REQUEST_FROM_PLATFORM_ORCH.value,
            request_obj=request_obj,
            request=request)

        if request.config_flags.get("isRagChainEnabled"):
            request = anonymize_request(request, __bot_id=request.bot_id,
                                        __ld_flag_name_for_default_disabled=config.PII_ANONYMIZATION_FLAG_NAME)

        request = await set_params_for_ans_request(request)
        if request.error.code == status.HTTP_400_BAD_REQUEST:
            request.status = APIStatus.FAILED.value
            request.timestamp = str(get_utc_timestamp_in_milliseconds())
            request.nlu_info.total_response_time_taken = get_time_difference_in_seconds(time1=time_in,
                                                                                        time2=time.time(),
                                                                                        round_off_decimals=3)
            logger.error(f"Error occurred while setting params from config for the request. "
                         f"Request {request.request_id} failed.")
            raise BadRequestException(request.error.message)

        if request.config_flags.get("isRagChainEnabled"):
            rephrased_query = await get_llm_query(bot_id=request.bot_id, request_id=request.request_id,
                                                  original_query=request.query,
                                                  chat_history=request.conversation_context_history,
                                                  rephrased_query=request.rephrased_query,
                                                  env=request.env)
            request.rephrased_query = rephrased_query
        else:
            request.rephrased_query = request.query if request.query else request.rephrased_query
        logger.info(f"llm response generation request received with {request.request_id=} {request.bot_id=} {request.query=}")

        final_response = await stream_llm_response(request=request)
        await produce_llm_ans_event_in_conversation_tracker(
            state=ConversationTrackerEvents.DS_GATEWAY_ANS_API_SENT_RESPONSE_TO_PLATFORM_ORCH.value,
            request_obj=request_obj,
            request=request)
        logger.info(f"request final output json: {final_response}")

    except Exception as ex:
        if request.config_flags.get("isLLMVaEnabled"):
            produce_error_in_kafka(exception=DSExceptions, request_id=request.request_id, conversation_id=request.conversation_id,
                                   bot_id=request.bot_id, bot_ref_id=request.bot_ref_id)
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1]
        logger.exception(f"Error occurred during llm prediction. Error : {type(ex)} : {str(ex)}", exc_info=True)
    finally:
        logger.info(f"Time taken for llm {request_obj.url.path} response is {time.time() - time_in} seconds")



# This will be deprecated once orchestrator starts supporting streaming response for this API.
# The idea is to keep a single logic for streaming responses
async def stream_llm_response(request: LLMAnsRequest):
    response_topic_name = request.call_back_topic or config.KAFKA_TOPIC_NAME
    time_in = time.time()
    bot_id = request.bot_id
    bot_ref_id = request.bot_ref_id
    conversation_id = request.conversation_id
    request_id = request.request_id

    try:
        logger.info(f"request input json: {request.model_dump_json(by_alias=True)}")
        log_id = f"Bot_id:{request.bot_id}, conversation_id:{request.conversation_id} :"

        executor, user_profile, is_rag_chain_disabled, has_unclassified_instructs, _ = await get_executor_and_user_profile(request=request)

        # init verdict and state
        max_verdict_check_length = config.MAX_AGENT_VERDICT_CHECK_LENGTH if is_rag_chain_disabled else config.MAX_VERDICT_CHECK_LENGTH
        share_unclassified_response = request.prompt.share_unclassified_response
        process_response = True if has_unclassified_instructs else False
        response_processor = AgentResponseProcessor(process_response) if is_rag_chain_disabled else LLMResponseProcessor(process_response)
        verdict_value = VerdictType.ANSWERABLE.value if (process_response is False) else None
        state = StreamingState.VERDICT_EXTRACTED.value if verdict_value else StreamingState.WAITING_FOR_VERDICT.value
        token_buffer = ""

        async for stream_object in executor.run_streamer(user_profile):
            request.nlu_info.rephrased_query = stream_object.rephrased_query
            token_content = stream_object.token_content
            token_buffer += token_content

            if state == StreamingState.WAITING_FOR_VERDICT.value:
                verdict_match, verdict_value = response_processor.extract_verdict(token_buffer)
                if verdict_match:
                    state = StreamingState.VERDICT_EXTRACTED.value
                    token_content = response_processor.remove_verdict_info(token_buffer)
                elif len(token_buffer) >= max_verdict_check_length:
                    if not is_rag_chain_disabled:
                        verdict_value = VerdictType.UNCLASSIFIED.value
                        share_unclassified_response = False
                        state = StreamingState.VERDICT_EXTRACTED.value
                        logger.error(f"Unable to extract the verdict_value within the token_buffer : {token_buffer}. Setting verdict_value to unclassified")
                        break
                    else:
                        verdict_value = VerdictType.ANSWERABLE.value
                        state = StreamingState.VERDICT_EXTRACTED.value
                        token_content = response_processor.remove_verdict_info(token_buffer)
                        logger.info(f"Unable to extract the verdict_value within the token_buffer : {token_buffer}. Setting verdict_value to answerable since response generation is from Agent")

            if state == StreamingState.VERDICT_EXTRACTED.value:
                if verdict_value == VerdictType.UNCLASSIFIED.value and share_unclassified_response is False:
                    state = StreamingState.NOT_STREAMING.value
                    break
                else:
                    state = StreamingState.STREAMING_TOKENS.value

            if state == StreamingState.STREAMING_TOKENS.value:
                output_content, response_token_idx = response_processor.process_streaming_data(token_content)
                if output_content:
                    request.nlu_info.generated_answer += output_content
                    is_eos = True if executor.handover_to_orchestrator else False
                    if response_token_idx == 0:
                        logger.info(f"Time till first token logged is {time.time() - time_in} seconds")
                        request.token = Token(index=response_token_idx, is_eos=is_eos, text=output_content)
                        source_documents = stream_object.answer_source_documents
                        # request.nlu_info.answer_source_documents.extend(source_documents)
                        request.nlu_info.total_response_time_taken = get_time_difference_in_seconds(time1=time_in,
                                                                                                    time2=time.time(),
                                                                                                    round_off_decimals=3)
                        request.timestamp = str(get_utc_timestamp_in_milliseconds())
                        if not executor.handover_to_orchestrator:
                            kafka_util.produce_message(
                                topic=response_topic_name, key=conversation_id, value=request.create_response(),
                                log_message=True)
                        else:
                            request.create_response()
                    else:
                        request.token = Token(index=response_token_idx, is_eos=is_eos, text=output_content)
                        kafka_util.produce_message(
                            topic=response_topic_name, key=conversation_id, value=request.to_token_dict(),
                            log_message=False)
        # Mark end-of-stream token
        if not executor.handover_to_orchestrator:
            request.token = Token(index=stream_object.token_idx, is_eos=True, text="")

        # Attach the aggregated custom-fields list to the response one time (stringified for API response)
        if executor is not None and hasattr(executor, "action_custom_fields_response") and executor.action_custom_fields_response:
            request.custom_fields = _stringify_custom_fields(executor.action_custom_fields_response)
        if executor is not None and hasattr(executor, "tools_invoked") and executor.tools_invoked:
            request.tools_invoked = executor.tools_invoked
        request.nlu_info.response_type = response_processor.create_response_type(verdict_value)
        request.original_conversation_context_history.extend(extend_original_chat_history(query=request.query,
                                                                                          answer=request.nlu_info.generated_answer))
        request.nlu_info.total_response_time_taken = get_time_difference_in_seconds(time1=time_in,
                                                                                    time2=time.time(),
                                                                                    round_off_decimals=3)

        request.timestamp = str(get_utc_timestamp_in_milliseconds())
        request.status = APIStatus.SUCCESS.value
        request.error.code = ErrorCodes.SUCCESSFUL.value[0]
        request.error.message = ErrorCodes.SUCCESSFUL.value[1]
        response = request.create_response()

        final_output_log_text = ""
        if not is_rag_chain_disabled:
            logger.info(f"{log_id} request final stream_llm_response output json before PII DeAnonymization: {response}")
            response = deanonymize_rag_response(response, __bot_id=request.bot_id, __ld_flag_name_for_default_disabled=config.PII_ANONYMIZATION_FLAG_NAME)
            final_output_log_text = " after PII DeAnonymization"
        logger.info(f"{log_id} request final stream_llm_response output json {final_output_log_text}: {response}")

        if not (request.nlu_info.response_type == ResponseType.UNCLASSIFIED.value and share_unclassified_response is False):
            kafka_util.produce_message(topic=response_topic_name, key=conversation_id, value=response)
            kafka_util.poll(timeout_seconds=0)
    except (LLMExceptions or DSExceptions) as e:
        if request.config_flags.get("isLLMVaEnabled"):
            produce_error_in_kafka(exception=e, request_id=request_id, conversation_id=conversation_id, bot_id=bot_id, bot_ref_id=bot_ref_id)
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1] + f" Error type: {e.type}"
        logger.exception(f"Error occurred during llm prediction. Error : {type(e)} : {str(e)}", exc_info=True)

    except Exception as e:
        if request.config_flags.get("isLLMVaEnabled"):
            produce_error_in_kafka(exception=DSExceptions, request_id=request_id, conversation_id=conversation_id, bot_id=bot_id, bot_ref_id=bot_ref_id)
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1]
        logger.exception(f"Error occurred during llm prediction. Error : {type(e)} : {str(e)}", exc_info=True)

    finally:
        kafka_util.flush_messages()
        return request



async def stream_llm_response_direct(request: LLMAnsRequest, request_obj: Request = None):
    response_topic_name = request.call_back_topic or config.KAFKA_TOPIC_NAME
    time_in = time.time()
    bot_id = request.bot_id
    bot_ref_id = request.bot_ref_id
    conversation_id = request.conversation_id
    request_id = request.request_id
    executor = None

    try:
        logger.info(f"request input json: {request.model_dump_json(by_alias=True)}")
        log_id = f"Bot_id:{request.bot_id}, conversation_id:{request.conversation_id} :"

        executor, user_profile, is_rag_chain_disabled, has_unclassified_instructs, is_conversation_filler_enabled = await get_executor_and_user_profile(request=request)

        # init verdict and state
        max_verdict_check_length = config.MAX_AGENT_VERDICT_CHECK_LENGTH if is_rag_chain_disabled else config.MAX_VERDICT_CHECK_LENGTH
        share_unclassified_response = request.prompt.share_unclassified_response
        share_unclassified_topic = request.prompt.share_unclassified_topic
        process_response = True if has_unclassified_instructs else False
        response_processor = AgentResponseProcessor(process_response) if is_rag_chain_disabled else LLMResponseProcessor(process_response)
        verdict_value = VerdictType.ANSWERABLE.value if (process_response is False) else None
        state = StreamingState.VERDICT_EXTRACTED.value if verdict_value else StreamingState.WAITING_FOR_VERDICT.value
        token_buffer = ""
        time_in_llm_generation = time.time()
        async for stream_object in executor.run_streamer(user_profile):
            # directly stream filler message
            if isinstance(stream_object, FillerObject):
                request.token = Token(index=stream_object.filler_idx, is_eos=stream_object.is_eos,
                                      text=stream_object.filler_content)
                if is_conversation_filler_enabled:
                    # Parallel Pipeline, SSE Event
                    if request.stream_direct:
                        if not request.send_full_answer:
                            yield request.to_filler_token_dict()
                    # Merge Pipeline, Kafka
                    else:
                        kafka_util.produce_message(
                            topic=response_topic_name,
                            key=conversation_id,
                            value=request.to_filler_token_dict(),
                            log_message=False
                        )
                continue

            request.nlu_info.rephrased_query = stream_object.rephrased_query
            token_content = stream_object.token_content
            token_buffer += token_content

            if state == StreamingState.WAITING_FOR_VERDICT.value:
                verdict_match, verdict_value = response_processor.extract_verdict(token_buffer)
                if verdict_match:
                    state = StreamingState.VERDICT_EXTRACTED.value
                    token_content = response_processor.remove_verdict_info(token_buffer)
                elif len(token_buffer) >= max_verdict_check_length:
                    if not is_rag_chain_disabled:
                        verdict_value = VerdictType.UNCLASSIFIED.value
                        share_unclassified_response = False
                        state = StreamingState.VERDICT_EXTRACTED.value
                        logger.error(f"Unable to extract the verdict_value within the token_buffer : {token_buffer}. Setting verdict_value to unclassified")
                        break
                    else:
                        verdict_value = VerdictType.ANSWERABLE.value
                        state = StreamingState.VERDICT_EXTRACTED.value
                        token_content = response_processor.remove_verdict_info(token_buffer)
                        logger.info(f"Unable to extract the verdict_value within the token_buffer : {token_buffer}. Setting verdict_value to answerable since response generation is from Agent")

            if state == StreamingState.VERDICT_EXTRACTED.value:
                if verdict_value == VerdictType.UNCLASSIFIED.value and share_unclassified_response is False:
                    state = StreamingState.NOT_STREAMING.value
                    continue
                else:
                    state = StreamingState.STREAMING_TOKENS.value

            if state == StreamingState.STREAMING_TOKENS.value:
                output_content, response_token_idx = response_processor.process_streaming_data(token_content)
                if output_content:
                    request.nlu_info.generated_answer += output_content
                    is_eos = True if executor.handover_to_orchestrator else False
                    if response_token_idx == 0:
                        first_token_latency = time.time() - request.time_in
                        if request.eval_request:
                            first_token_latency = time.time() - time_in_llm_generation
                        request_timing = request_timing_context.get()
                        request_timing['first_token_latency'] = first_token_latency*1000  # convert to ms    
                        logger.info(f"Time till first token logged is {first_token_latency} seconds")
                        request.token = Token(index=response_token_idx, is_eos=is_eos, text=output_content)
                        source_documents = stream_object.answer_source_documents
                        # request.nlu_info.answer_source_documents.extend(source_documents)
                        request.nlu_info.total_response_time_taken = get_time_difference_in_seconds(time1=time_in,
                                                                                                    time2=time.time(),
                                                                                                    round_off_decimals=3)
                        
                        # calculate llm generation latency. Tool latency should be calculated here
                        # to estimate the llm generation latency correctly.
                        tool_runtime_list = []
                        for entry in list(request_timing.keys()):
                            if entry.startswith('tool_function_'):
                                runtime = request_timing.pop(entry)
                                tool_runtime_list.extend(runtime)
                                request_timing[entry] = total_unique_runtime(runtime) * 1000 # convert to ms
                        total_tool_latency = total_unique_runtime(tool_runtime_list)*1000 # convert to ms
                        llm_generation_latency = (time.time() - time_in_llm_generation)*1000 - total_tool_latency  # convert to ms
                        request_timing['llm_generation_latency'] = llm_generation_latency
                        request_timing['total_tool_latency'] = total_tool_latency
                        request.timestamp = str(get_utc_timestamp_in_milliseconds())
                        if not executor.handover_to_orchestrator:
                            if request.stream_direct:
                                if not request.send_full_answer:
                                    yield request.create_response()
                            else:
                                kafka_util.produce_message(
                                    topic=response_topic_name, key=conversation_id, value=request.create_response(),
                                    log_message=True)
                            await produce_llm_ans_event_in_conversation_tracker(
                                state=ConversationTrackerEvents.DS_GATEWAY_ANS_API_SENT_FIRST_CHUNK_TO_ORCHESTRATOR.value,
                                request_obj=request_obj,
                                request=request)
                        else:
                            request.create_response()
                    else:
                        request.token = Token(index=response_token_idx, is_eos=is_eos, text=output_content)
                        if request.stream_direct:
                            if not request.send_full_answer:
                                yield request.to_token_dict()
                        else:
                            kafka_util.produce_message(
                                topic=response_topic_name, key=conversation_id, value=request.to_token_dict(),
                                log_message=False)
        if not executor.handover_to_orchestrator:
            request.token = Token(index=stream_object.token_idx, is_eos=True, text="")
        last_token_latency = time.time() - request.time_in
        if request.eval_request:
            try:
                request.llm_intermediate_steps = getattr(executor, "eval_intermediate_events", [])
                logger.info(f"Added intermediate events metadata for evaluation experiment.")
            except Exception as e:
                logger.debug(f"Couldn't get intermediate steps for evaluation : {e}")
                pass
            last_token_latency = time.time() - time_in_llm_generation
        request_timing = request_timing_context.get()
        request_timing['last_token_latency'] = last_token_latency*1000  # convert to ms
        if executor is not None and hasattr(executor, "action_custom_fields_response") and executor.action_custom_fields_response:
            request.custom_fields = _stringify_custom_fields(executor.action_custom_fields_response)
        if executor is not None and hasattr(executor, "tools_invoked") and executor.tools_invoked:
            request.tools_invoked = executor.tools_invoked

        request.nlu_info.response_type = response_processor.create_response_type(verdict_value)
        request.original_conversation_context_history.extend(extend_original_chat_history(query=request.query,
                                                                                          answer=request.nlu_info.generated_answer))
        request.nlu_info.total_response_time_taken = get_time_difference_in_seconds(time1=time_in,
                                                                                    time2=time.time(),
                                                                                    round_off_decimals=3)

        request.timestamp = str(get_utc_timestamp_in_milliseconds())
        request.status = APIStatus.SUCCESS.value
        request.error.code = ErrorCodes.SUCCESSFUL.value[0]
        request.error.message = ErrorCodes.SUCCESSFUL.value[1]
        # set unclassified intent logic for voice pipeline in final response to be shared
        if request.nlu_info.response_type == ResponseType.UNCLASSIFIED.value and share_unclassified_topic:
            request.trigger_intent = [{"intentName": VerdictType.UNCLASSIFIED.value}]
        response = request.create_response()

        final_output_log_text = ""
        if not is_rag_chain_disabled:
            logger.info(f"{log_id} request final stream_llm_response output json before PII DeAnonymization: {response}")
            response = deanonymize_rag_response(response, __bot_id=request.bot_id, __ld_flag_name_for_default_disabled=config.PII_ANONYMIZATION_FLAG_NAME)
            final_output_log_text = " after PII DeAnonymization"
        logger.info(f"{log_id} request final stream_llm_response output json {final_output_log_text}: {response}")

        if not (request.nlu_info.response_type == ResponseType.UNCLASSIFIED.value and share_unclassified_response is False):
            if request.stream_direct:
                if request.nlu_info.response_type != ResponseType.UNCLASSIFIED.value:
                    yield response
            else:
                kafka_util.produce_message(topic=response_topic_name, key=conversation_id, value=response)
                kafka_util.poll(timeout_seconds=0)
        await produce_llm_ans_event_in_conversation_tracker(
                state=ConversationTrackerEvents.DS_GATEWAY_ANS_API_SENT_LAST_CHUNK_TO_ORCHESTRATOR.value,
                request_obj=request_obj,
                request=request)
    except (LLMExceptions or DSExceptions) as e:
        if request.config_flags.get("isLLMVaEnabled"):
            if request.stream_direct:
                error_message = ResponseStreamErrorMessage(
                    triggerType="EVENT",
                    eventType="ERROR",
                    subType=e.type,
                    requestId=request_id,
                    conversationId=conversation_id,
                    botId=bot_id,
                    botRefId=bot_ref_id
                )
                yield error_message.to_json()
            else:
                produce_error_in_kafka(exception=e, request_id=request_id, conversation_id=conversation_id, bot_id=bot_id, bot_ref_id=bot_ref_id)
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1] + f" Error type: {e.type}"
        logger.exception(f"Error occurred during llm prediction. Error : {type(e)} : {str(e)}", exc_info=True)

    except Exception as e:
        if request.config_flags.get("isLLMVaEnabled"):
            if request.stream_direct:
                error_message = ResponseStreamErrorMessage(
                    triggerType="EVENT",
                    eventType="ERROR",
                    subType=DSExceptions.type,
                    requestId=request_id,
                    conversationId=conversation_id,
                    botId=bot_id,
                    botRefId=bot_ref_id
                )
                yield error_message.to_json()
            else:
                produce_error_in_kafka(exception=DSExceptions, request_id=request_id, conversation_id=conversation_id, bot_id=bot_id, bot_ref_id=bot_ref_id)
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1]
        logger.exception(f"Error occurred during llm prediction. Error : {type(e)} : {str(e)}", exc_info=True)

    except BaseException as e:
        if request.config_flags.get("isLLMVaEnabled"):
            if request.stream_direct:
                error_message = ResponseStreamErrorMessage(
                    triggerType="EVENT",
                    eventType="ERROR",
                    subType=DSExceptions.type,
                    requestId=request_id,
                    conversationId=conversation_id,
                    botId=bot_id,
                    botRefId=bot_ref_id
                )
                yield error_message.to_json()
            else:
                produce_error_in_kafka(exception=DSExceptions, request_id=request_id, conversation_id=conversation_id, bot_id=bot_id, bot_ref_id=bot_ref_id)
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1]
        logger.exception(f"Error occurred during llm prediction. Error : {type(e)} : {str(e)}", exc_info=True)

    finally:
        if not request.stream_direct:
            kafka_util.flush_messages()


async def get_llm_sync_response(request: LLMAnsRequest) -> LLMAnsRequest:
    try:
        bot_id = request.bot_id
        request_id = request.request_id
        query = request.query
        logger.info(f"llm sync response generation request received with {request_id=} {bot_id=} {query=}")
        executor, user_profile, is_rag_chain_disabled, has_unclassified_instructs, _ = await get_executor_and_user_profile(request=request)
        answer, source_documents, rephrased_query = await executor.sync_response(user_profile)
        process_response = True if has_unclassified_instructs else False
        response_processor = AgentResponseProcessor(process_response) if is_rag_chain_disabled else LLMResponseProcessor(process_response)
        if request.eval_request:
            try:
                request.llm_intermediate_steps = getattr(executor, "eval_intermediate_events", [])
                logger.info(f"Added intermediate events metadata for evaluation experiment.")
            except Exception as e:
                logger.debug(f"Couldn't get intermediate steps for evaluation : {e}")
                pass
        request.nlu_info.response_type, request.nlu_info.generated_answer = response_processor.extract_verdict_response(answer)
        if executor is not None and hasattr(executor, "action_custom_fields_response") and executor.action_custom_fields_response:
            request.custom_fields = _stringify_custom_fields(executor.action_custom_fields_response)
        if executor is not None and hasattr(executor, "tools_invoked") and executor.tools_invoked:
            request.tools_invoked = executor.tools_invoked
        if request.nlu_info.response_type == ResponseType.UNCLASSIFIED.value and request.prompt.share_unclassified_response is False:
            request.nlu_info.generated_answer = ""
        request.token.is_eos = True
        request.token.text = request.nlu_info.generated_answer
        request.nlu_info.rephrased_query = rephrased_query
        # request.nlu_info.answer_source_documents.extend(source_documents)
        request.original_conversation_context_history.extend(extend_original_chat_history(query=request.query,
                                                                                          answer=request.nlu_info.generated_answer))
        request.status = APIStatus.SUCCESS.value
        request.error.code = ErrorCodes.SUCCESSFUL.value[0]
        request.error.message = ErrorCodes.SUCCESSFUL.value[1]

    except (LLMExceptions or DSExceptions) as e:
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1] + f" Error type: {e.type}"
        logger.exception(f"Error occurred during llm prediction. Error : {type(e)} : {str(e)}", exc_info=True)

    except Exception as e:
        request.status = APIStatus.FAILED.value
        request.error.code = ErrorCodes.INTERNAL_SERVER_ERROR.value[0]
        request.error.message = ErrorCodes.INTERNAL_SERVER_ERROR.value[1]
        logger.exception(f"Error occurred during llm prediction. Error : {type(e)} : {str(e)}", exc_info=True)

    finally:
        return request


async def get_executor_and_user_profile(request: LLMAnsRequest):
    """
            Initialize and return the appropriate executor (RAG Chain or Agent) along with user profile and configuration flags.

            Args:
                request: The LLM answer request containing bot configuration and user query

            Returns:
                tuple: (executor, user_profile, is_rag_chain_disabled, has_unclassified_instructs, is_conversation_filler_enabled)
    """
    # Extract basic request information
    bot_id = request.bot_id
    bot_ref_id = request.bot_ref_id
    conversation_id = request.conversation_id

    request_id = request.request_id
    message_id = request.message_id
    visitor_info = request.visitor
    filter_tags = request.filter_tags if request.filter_tags else {}
    chat_history = request.conversation_context_history
    llm_name = request.llm_name
    user = request.user
    query = request.rephrased_query
    processed_utterance = request.processed_utterance
    mode = request.env
    is_response_analyser_request = request.is_response_analyser_request
    tags = [bot_id, conversation_id, llm_name, request_id]
    is_mcp_enabled = request.config_flags.get("IS_MCP_ENABLED", False)
    user_profile = user.get("profile", []) if user else []
    is_conversation_filler_enabled = False  # This will only be true for Agent framework and not for RAG framework
    is_agent_assist_enabled = request.config_flags.get("isAgentAssistEnabled")
    is_knowledge_tool_enabled = False
    is_agent_assist_copilot_enabled = request.config_flags.get("IsAgentAssistCopilotEnabled")
    is_rag_chain_disabled = not request.config_flags.get("isRagChainEnabled")


    llm_capability_model_config = await get_llm_config_models_by_capability(request=request)

    if is_rag_chain_disabled or is_agent_assist_copilot_enabled:
        framework_type = ResponseGenerationFramework.AGENT.value
        algorithm = request.model.model_id = llm_capability_model_config.agent_model_config.algorithm
    else:
        framework_type = ResponseGenerationFramework.RAG.value
        algorithm = request.model.model_id = llm_capability_model_config.rag_model_config.algorithm

    if is_agent_assist_copilot_enabled:
        capability_task = CapabilityTasks.AGENT_ORCHESTRATOR.value
        capability = Capability.AGENT_ASSIST_COPILOT.value[1]
    else:
        capability_task = CapabilityTasks.AGENT_ORCHESTRATOR.value if is_rag_chain_disabled else CapabilityTasks.RESPONSE_GENERATOR.value
        if is_agent_assist_enabled:
            capability_task = getattr(CapabilityTasks,f"{'AGENT_ASSIST_' + capability_task}").value
        capability = Capability.RESPONSE_GENERATION.value[1]

    logger.info(f"Running on {framework_type} framework with model {request.model.model_id}")

    if request.stream_direct:
        request.config_flags["isMultiLangForAnswerAIEnabled"] = True

    query_for_retrival = query
    if processed_utterance and request.config_flags.get("isMultiLangForAnswerAIEnabled") and \
            request.config_flags.get("sourceUserQueryLanguage").lower() != config.DEFAULT_USER_LANGUAGE:
        query_for_retrival = processed_utterance

    logger.info(f"Config flags status for bot-id: {bot_id} is: {request.config_flags}")

    # Fetch the prompt data, RAG/AGENT-specific unclassified instruction flag, config variables etc
    (prompt_data_obj, has_unclassified_instructs, prompt_config_vars,
     is_rte_enabled_agent, user_profile) = await get_formatted_prompts(request_obj=request, capability_task=capability_task,
                                                                       algorithm=algorithm, framework_type=framework_type,
                                                                       user_profile=user_profile, capability=capability)

    is_agent_framework = framework_type == ResponseGenerationFramework.AGENT.value
    is_advanced_series_model = algorithm in [AdvancedSeriesModels.NETOMI_AGENTIC_OS_MODEL_V7_1.value, AdvancedSeriesModels.NETOMI_AGENTIC_OS_MODEL_V8.value]
    use_advanced_model_series = bool(is_agent_framework and is_advanced_series_model)
    logger.info(f"the advanced series flag is set to {use_advanced_model_series}")

    if use_advanced_model_series:
        try:
            if not prompt_config_vars.get(PromptBotConfigs.TOOL_HISTORY_CONTEXT_MANAGEMENT.value, False):
                logger.info(
                    f"Advanced-series enabled: forcing {PromptBotConfigs.TOOL_HISTORY_CONTEXT_MANAGEMENT.value}=True"
                )
            prompt_config_vars[PromptBotConfigs.TOOL_HISTORY_CONTEXT_MANAGEMENT.value] = True
        except Exception as e:
            logger.warning(f"Failed to force tool history context management for advanced-series: {e}")

    # setup agent framework
    ai_agent_framework = prompt_config_vars.get(PromptBotConfigs.AI_AGENT_FRAMEWORK.value, AiAgentFramework.LANGGRAPH_V1.value)
    ai_agent_framework = AiAgentFramework.LANGGRAPH_V1.value if is_knowledge_tool_enabled or algorithm != DefaultAgenticOSModel.NETOMI_AGENTIC_OS_MODEL_V1.value else ai_agent_framework

    # Enable tool calling
    tool_history_context_enabled = False if request.human_agent_instruction.get("target_user") == CopilotTargetUser.AGENT.value\
        else prompt_config_vars.get(PromptBotConfigs.TOOL_HISTORY_CONTEXT_MANAGEMENT.value, True)

    include_tool_call = True if ai_agent_framework == AiAgentFramework.LANGGRAPH_V1.value and tool_history_context_enabled else False
    if use_advanced_model_series:
        include_tool_call = True

    # Formulate Chat history
    chat_history = format_chat_history(chat_history=chat_history,
                                       is_rag_chain_enabled=request.config_flags.get("isRagChainEnabled"),
                                       include_tool_call=include_tool_call)

    # RAG Executor
    if framework_type == ResponseGenerationFramework.RAG.value:
        try:
            max_retrievable_documents = int(prompt_config_vars.get(PromptBotConfigs.MAX_RETRIEVABLE_DOCUMENTS.value, 4))
        except (ValueError, TypeError):
            max_retrievable_documents = config.TOTAL_DOC_RETRIEVE_FOR_RESPONSE_GEN
        executor = RagChainResponseGeneration(bot_id=bot_id, mode=mode, llm_name=llm_name, query=query,
                                              query_for_retrival=query_for_retrival,
                                              request_id=request_id, chat_history=chat_history,
                                              prompt_data=prompt_data_obj, filter_tags=filter_tags,
                                              max_retrievable_documents=max_retrievable_documents,
                                              llm_capability_model_config=llm_capability_model_config,
                                              conversation_id=conversation_id)

        logger.info(f"Using RAG Chain Response Generation for bot-id: {bot_id} and query: {query}")

    # Agent Executor
    else:
        is_conversation_filler_enabled = request.send_full_answer is False and prompt_config_vars.get(PromptBotConfigs.SHARE_CONVERSATION_FILLER_MESSAGE.value, False)
        is_thinking_enabled = prompt_config_vars.get(PromptBotConfigs.IS_THINKING_ENABLED.value, False)
        executor = await LanggraphAgent.create(bot_id=bot_id, bot_ref_id=bot_ref_id, user=user_profile, conversation_id=conversation_id,
                                             message_id=message_id, mode=mode, llm_name=llm_name, query=query,
                                             request_id=request_id, visitor_info=visitor_info, chat_history=chat_history, tags=tags,
                                             prompt_data=prompt_data_obj, prompt_config_vars=prompt_config_vars,
                                             is_rte_enabled_agent=is_rte_enabled_agent, filter_tags=filter_tags,
                                             has_topic_instructs=has_unclassified_instructs, is_mcp_enabled=is_mcp_enabled,
                                             is_conversation_filler_enabled=is_conversation_filler_enabled,
                                             is_response_analyser_request=is_response_analyser_request,
                                             has_unclassified_instructs=has_unclassified_instructs,
                                             is_agent_assist_enabled=is_agent_assist_enabled,
                                             is_knowledge_tool_enabled=is_knowledge_tool_enabled,
                                             is_thinking_enabled=is_thinking_enabled,
                                             is_advanced_model_series_enabled=use_advanced_model_series,
                                             llm_capability_model_config=llm_capability_model_config,
                                             is_agent_assist_copilot_enabled=is_agent_assist_copilot_enabled,
                                             user_and_human_agent_chat_history=request.user_and_human_agent_chat_history,
                                             human_agent_instruction=request.human_agent_instruction,
                                            config_flags=request.config_flags)


        logger.info(f"Using {LanggraphAgent.__name__} Agent Response Generation for bot-id: {bot_id} and query: {query}")

    return executor, user_profile, is_rag_chain_disabled, has_unclassified_instructs, is_conversation_filler_enabled
