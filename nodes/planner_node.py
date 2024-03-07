from langchain_experimental.llms.ollama_functions import OllamaFunctions 
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field 
from typing import List



class Plan(BaseModel):
    """Plan to follow in the future"""
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")
    


planner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

{objective}"""
)

chat_model = OllamaFunctions(model='mistral:7b-instruct-v0.2-q6_K', temperature = 0.1)

planner = create_structured_output_runnable(
    Plan, chat_model, planner_prompt, mode='openai-functions'
)