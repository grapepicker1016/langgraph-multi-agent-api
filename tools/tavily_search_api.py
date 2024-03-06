import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults

tavily = TavilySearchResults(max_results=3)