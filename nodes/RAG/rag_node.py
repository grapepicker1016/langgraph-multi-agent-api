import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
import os


from langchain_community.vectorstores import Weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatOllama

from langchain.prompts import PromptTemplate

import weaviate
from langchain.globals import set_llm_cache
from langchain.cache import RedisCache
import redis

from typing import Callable
from langchain.prompts import StringPromptTemplate
import re
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    StructuredChatAgent
)
from typing import Union, Optional
from langchain.schema import AgentAction, AgentFinish
from langchain.output_parsers import OutputFixingParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.schema.output_parser import StrOutputParser


# search = TavilySearchResults(max_results=1,
#                              description=
#         "A search engine"
#         "Useful for when you need to answer questions about current events. use this tool to search on questions which you don't have answer."
#         "Input should be a search query."
#     )



REDIS_URL = "redis://localhost:6379/0"

redis_client = redis.Redis.from_url(REDIS_URL)
set_llm_cache(RedisCache(redis_client))


message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", session_id="my-session", ttl=600
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)


client = weaviate.Client(
  url="http://localhost:8080",
)

vectorstore = Weaviate(client, 
                       "GRP", 
                       "content")

retriever = vectorstore.as_retriever()

# llm = ChatOllama(model="llama2:13b-chat")


retriever_tool = create_retriever_tool(
    retriever,
    "mediwave_search",
    "Search any information only about mediwave or mindwave. For any questions related to Mediwave or mindwave, you must use this tool!",
)
# -------------------- General Conversation Tool -----------
#general conversation
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, below provided the chat history of the AI and Human make use of it for context reference. if the question is standalone then provide Answer to the question on your own. make sure it sounds like human and official assistant:
            \n"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
    
_model = ChatOllama(model="llama2:7b-chat")






def general_conv(query: str):
    """General conversation, Use this tool to make any general conversation with the user."""
    

    chain5 = _prompt | _model | StrOutputParser()
    
    result = chain5.invoke({"input": query, 'chat_history': memory.buffer_as_messages})
    
    
    return result


conversation_tool = StructuredTool.from_function(
    func=general_conv,
    name="gen_conv",
    description="useful for when you need to answer general question or conversations General conversation, Use this tool to make any general conversation or greeting with the user. and provide the actual user input to this tool ",
    return_direct=True
    # coroutine= ... <- you can specify an async method if desired as well
)

# ------------------- General Conv -------------------------

tools = [search, retriever_tool, conversation_tool]


# if the question is any related or referenced the chat history entities make use of that to answer the the given question effectively.don't use chat history if the question is standalone abd make sure to answer the given question alone. if no chat history is provided then continue with the question alone. if it is a normal conversation then provide the answer as Final Answer and don't need to use any tool.

# chat_history: \n {chat_history}
# End of chat history.


# if the question is any related or referenced the chat history entities make use of that to answer the the given question effectively.don't use chat history if the question is standalone abd make sure to answer the given question alone. if no chat history is provided then continue with the question alone. if it is a normal conversation then provide the answer as Final Answer and don't need to use any tool.

# chat_history: \n {chat_history}
# End of chat history.

template = """You are an AI assistant, Answer the following question as best you can. if the user is making a general conversation/greeting or normal conversation then use the 'gen_conv' tool.  

You have access to the following tools. make sure to use the format, if you don't know the answer then just say I don't know as Final answer. if anything went wrong just inform the user like try again or try after sometime:

if the question is any related or referenced the chat history entities make use of that to obtain the context to answer the the given question effectively. if the given question is a followup question, then find the context and then select the appropriate tool for it. don't use chat history if the question is standalone and make sure to answer the given question alone. if no chat history is provided then continue with the question alone. 

chat_history: \n {chat_history}
End of chat history.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times if needed if you know the answer on your own then skip the Action and Action Input )
Thought: I now know the final answer
Final Answer: the final answer to the original input question. if you know the Final answer in the beginning then give the Final Answer there is no need to give Thought or Observation. 

Begin! and Strict to the Format. always put 'Final Answer' at the beginning of the final answer. and use the tool name as same as given.

Question: {input}
Thought:{agent_scratchpad}"""



# template = """You are an AI assistant, you can make normal conversations like humans, Answer the following questions as best you can. You have access to the following tools and don't need to use the tools while making normal conversations. make sure to use the format, if you don't know the answer just say I don't know as Final answer. if anything went wrong just inform the user like try again or try after sometime:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question. make sure to put 'Final Answer' at the beginning of the final answer

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}"""



class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: list

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools_getter=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "chat_history"],)



class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

output_parser = CustomOutputParser()


from langchain.chains import LLMChain

model = ChatOllama(model='mistral:7b-instruct-v0.2-q6_K')
# model = ChatOllama(model='falcon:40b-instruct-q4_1')
# model = ChatOllama(model='llama2:13b-chat-q8_0')


llm_chain = LLMChain(llm=model, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)
# prompt = hub.pull("hwchase17/structured-chat-agent")


# agent = create_structured_chat_agent(
#     llm=model,
#     prompt=prompt,
#     # output_parser=output_parser,
#     # stop=["\nObservation:"],
#     tools=tools,
# # )
from operator import itemgetter

from langchain.globals import set_debug

set_debug(True)
from langchain.globals import set_verbose

set_verbose(True)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory, 
    # return_only_outputs=True

) 

# | {'output' : lambda x : itemgetter(x['output'])}



# Add typing for input
class Question(BaseModel):
    __root__: str


chain = agent_executor.with_types(input_type=Question)
