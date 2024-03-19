from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.schema.output_parser import StrOutputParser


#### ROUTER
# This is the router - responsible for chosing what to do
chain = PromptTemplate.from_template("""Given the user question below, classify it as either being about `weather` or `other`.
                                     
Do not respond with more than one word.

<question>
{question}
</question>

Classification:""") | ChatAnthropic() | StrOutputParser()

#### Agent
# Defint the agent, which one branch of the router will use
from langchain.agents import XMLAgent, tool, AgentExecutor
from langchain.chat_models import ChatAnthropic

model = ChatAnthropic(model="claude-2")

@tool
def search(query: str) -> str:
    """Search things about current events."""
    return "32 degrees"

tool_list = [search]

# Get prompt to use
prompt = XMLAgent.get_default_prompt()

# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


agent = (
    {
        "question": lambda x: x["question"],
        "intermediate_steps": lambda x: convert_intermediate_steps(x["intermediate_steps"])
    }
    | prompt.partial(tools=convert_tools(tool_list))
    | model.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgent.get_default_output_parser()
)

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

#### General chain
# Define a general chain, which will be used in other cases
general_chain = PromptTemplate.from_template("""Respond to the following question:

Question: {question}
Answer:""") | ChatAnthropic()

#### Router
# Define the routing logic
from langchain.schema.runnable import RunnableBranch

branch = RunnableBranch(
  (lambda x: "weather" in x["topic"].lower(), agent_executor),
  general_chain
)

#### All together!
# Put it all together now
full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch

full_chain.invoke({"question":"whats the weather in SF"})