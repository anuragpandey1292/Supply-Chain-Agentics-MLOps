from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import profiler_node, feature_engineering_node, inference_node

def build_supply_chain_workflow():
    """
    Builds and compiles the LangGraph multi-agent workflow for the supply chain copilot.
    """
    # Initialize StateGraph with our AgentState schema
    workflow = StateGraph(AgentState)

    # 1. Register the Agent Nodes
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("feature_engineer", feature_engineering_node)
    workflow.add_node("inference", inference_node)

    # 2. Define the Execution Flow (Edges)
    workflow.set_entry_point("profiler")
    workflow.add_edge("profiler", "feature_engineer")
    workflow.add_edge("feature_engineer", "inference")
    workflow.add_edge("inference", END)

    # 3. Compile into an executable agent application
    app = workflow.compile()
    return app

if __name__ == "__main__":
    # Test run the multi-agent pipeline locally
    app = build_supply_chain_workflow()
    
    initial_state = {
        "file_path": "data/train.csv",
        "profiler_report": None,
        "engineered_path": None,
        "prediction_input": None,
        "prediction_result": None,
        "current_step": "initialized",
        "error": None
    }
    
    print("🚀 Triggering Autonomous Multi-Agent Pipeline...")
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("🏁 PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"Final Step Reached: {final_state.get('current_step')}")
    print(f"PDF Health Report: {final_state.get('profiler_report', {}).get('pdf_path')}")
    print(f"Engineered Dataset: {final_state.get('engineered_path')}")
    print(f"Model Prediction Output: {final_state.get('prediction_result')}")
    print("="*50)