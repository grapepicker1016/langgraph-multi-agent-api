from langgraph.graph import MessageGraph, END
from typing import Dict
from .plan_and_schedule import plan_and_schedule
from .joiner import joiner 
from typing import Any, Union, Iterable, List, Tuple, Dict
from langchain_core.messages import AIMessage, BaseMessage


graph_builder = MessageGraph()

# 1.  Define vertices
# We defined plan_and_schedule above already
# Assign each node to a state variable to update
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)


## Define edges
graph_builder.add_edge("plan_and_schedule", "join")

### This condition determines looping logic


def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"


graph_builder.add_conditional_edges(
    start_key="join",
    # Next, we pass in the function that will determine which node is called next.
    condition=should_continue,
)
graph_builder.set_entry_point("plan_and_schedule")

chain = graph_builder.compile()