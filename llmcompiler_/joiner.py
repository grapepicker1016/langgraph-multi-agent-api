from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_functions import create_structured_output_runnable
from helpers.structured_output_runnable_ollama_functions import create_structured_output_runnable_custom
from langchain_core.messages import AIMessage
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from typing import Any, Union, Iterable, List, Tuple, Dict
from langchain import hub



class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples=""
)  # You can optionally add examples

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

joiner_llm = OllamaFunctions(model='mistral:7b-instruct-v0.2-q8_0')

# runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)


runnable = create_structured_output_runnable_custom(JoinOutputs, joiner_llm, joiner_prompt)

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return response + [
            SystemMessage(
                content=f"Context from last attempt: {decision.action.feedback}"
            )
        ]
    else:
        return response + [AIMessage(content=decision.action.response)]


def select_recent_messages(messages: list) -> dict:
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


joiner = select_recent_messages | runnable | _parse_joiner_output