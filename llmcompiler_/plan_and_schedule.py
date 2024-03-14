import itertools
from langchain_core.runnables import (
    chain as as_runnable,
)
from typing import List, Optional
from langchain_core.messages import BaseMessage
from .planner import planner
from .task_fetcher import schedule_tasks


@as_runnable
def plan_and_schedule(messages: List[BaseMessage], config):
    tasks = planner.stream(messages, config)
    # Begin executing the planner immediately
    tasks = itertools.chain([next(tasks)], tasks)
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        },
        config,
    )
    return scheduled_tasks