from vllm import LLM

llm = LLM(model=...)  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))
