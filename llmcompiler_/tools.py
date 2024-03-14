from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

from LLMCompiler.math_tools import get_math_tool

# tool_llm = OllamaFunctions(model='mistral:7b-instruct-v0.2-q8_0')

# calculate = get_math_tool(llm)

search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)

# tools = [search, calculate]
tools = [search]