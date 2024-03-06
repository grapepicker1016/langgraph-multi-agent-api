from langchain import hub 
from langchain_experimental.llms.ollama_functions import OllamaFunctions 

model = OllamaFunctions(model='mistral:7b-instruct-v0.2-q6_K')

prompt = hub.pull('hwchase17/openai-functions-agent')

model.bind(
    functions=[
    {'name': 'tavily_search_results_json', 
     'description': 'A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer     questions about current events. Input should be a search query.', 
     'parameters': {'type': 'object', 
                    'properties': 
                        {'query': {'description': 'search query to look up', 'type': 'string'}}, 
                        'required': ['query']}}
    ]
)

agent = model 
