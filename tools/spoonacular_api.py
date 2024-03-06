
from langchain.agents import AgentType, initialize_agent
from langchain.requests import Requests
from langchain_community.agent_toolkits import NLAToolkit
from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()
import os 

SPOONACULAR_API = os.getenv('SPOONACULAR_API_KEY')


llm = Ollama(model='mistral:7b-instruct-v0.2-q6_K')

requests = Requests(headers={"x-api-key": SPOONACULAR_API})

spoonacular_toolkit = NLAToolkit.from_llm_and_url(
    llm,
    "https://spoonacular.com/application/frontend/downloads/spoonacular-openapi-3.json",
    requests=requests,
    max_text_length=1800,  # If you want to truncate the response text
)

spoonacular_tools = spoonacular_toolkit.get_tools()

apis_to_remove = ['ByID', 'Information', 'Image', 'visualize', 'Analyzed', 'create', 'Website', 'Widget', 'Bulk', 'map', 'Week', 'delete', 'add', 'User', 'Videos', 'talk', 'Pairing', 'Restaurants', 'quickAnswer', 'ConversationSuggests', 'Templates', 'MenuItemSearch', 'SiteContent']

filtered_api_list = [item for item in spoonacular_tools if all(text not in item.name for text in apis_to_remove)]

openapi_format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: what to instruct the AI Action representative.
Observation: The Agent's response
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
Final Answer: the final answer to the original input question with the right amount of detail

When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response."""


mrkl = initialize_agent(
    filtered_api_list,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"format_instructions": openapi_format_instructions,},
    handle_parsing_errors=True
)

