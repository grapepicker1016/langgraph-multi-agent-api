from langchain import hub 
from langchain_experimental.llms.ollama_functions import OllamaFunctions 
from langchain_core.utils.function_calling import convert_to_openai_tool


model = OllamaFunctions(model='mistral:7b-instruct-v0.2-q6_K')

prompt = hub.pull('hwchase17/openai-functions-agent')































print(convert_to_openai_tool(TavilySearchResults(max_results=3)))