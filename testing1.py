from exceptions import LLMExceptions

if __name__ == "__main__":
    if LLMExceptions.type == "LLM_EXCEPTION":
        print("Error")
    if "LLM" in LLMExceptions.type:
        print("llm error")
