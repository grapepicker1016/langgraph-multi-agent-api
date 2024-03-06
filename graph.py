from langgraph.graph import StateGraph, END 
from state import PlanExecute

from nodes.replanner import replanner
from nodes.agent_api import agent 
from nodes.planner import planner
from nodes.replanner import Response






async def execute_step(state: PlanExecute):
    task = state['plan'][0]
    # agent_response = await final_chain.ainvoke({'input': task, 'chat_history': []})
    agent_response = await agent.ainvoke(task)

    return {
        "past_steps": (task, agent_response['agent_outcome'].return_values['output'])
    }
    
async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"objective": state['input']})
    return {'plan': plan.steps}

async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output, Response):
        return {'response': output.response}
    else:
        return {"plan": output.steps}

def should_end(state: PlanExecute):
    if state['response']:
        return True
    else:
        return False
    
# -----------------WORKFLOW -------------------------


workflow = StateGraph(PlanExecute)

#Add Plan Node 
workflow.add_node("planner", plan_step)

#Add the execution node 
workflow.add_node('agent', execute_step)

#Add Replan node
workflow.add_node('replan', replan_step)

#Set Entry Point 
workflow.set_entry_point('planner')

#From Plan we go to agent
workflow.add_edge('planner', 'agent')

#From Agent, we replan
workflow.add_edge('agent', 'replan')

workflow.add_conditional_edges(
    'replan',
    #Next, we pass in the funciton that will determine which node is to be called next
    should_end,
    {
        #if 'tools', then we call the tool node.
        True: END,
        False: 'agent',
    },
)

app = workflow.compile()