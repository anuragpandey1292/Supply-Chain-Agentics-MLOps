from typing import TypedDict, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Defines the shared memory state passed between agents in the LangGraph workflow.
    """
    file_path: str                             # Path to the uploaded raw dataset
    profiler_report: Optional[Dict[str, Any]]  # Results and PDF path from the Profiler Agent
    engineered_path: Optional[str]             # Path to the dataset after adaptive feature engineering
    prediction_input: Optional[Dict[str, Any]] # Payload prepared for model inference
    prediction_result: Optional[Dict[str, Any]]# Final prediction output from FastAPI/model hub
    current_step: str                          # Tracks the active phase of the pipeline
    error: Optional[str]                       # Captures any error messages encountered during execution   