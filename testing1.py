from exceptions import LLMExceptions

if LLMExceptions.type == "ERROR":
    print("Error")
if LLMExceptions.llm.value == "LLM":
    print("llm error")