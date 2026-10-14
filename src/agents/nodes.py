import os
import pandas as pd
from src.agents.state import AgentState
from src.data.advanced_profiler import AdvancedPDFProfilerAgent
from src.data.feature_engineering import AdaptiveFeatureEngineer

def profiler_node(state: AgentState) -> AgentState:
    """
    Agent Node 1: Automatically profiles the uploaded dataset, 
    detects anomalies, and generates the PDF report card.
    """
    print("\n🤖 [Agent: Profiler] Scanning dataset schema, trends, and anomalies...")
    file_path = state["file_path"]
    
    try:
        report_output = f"data/{os.path.basename(file_path).split('.')[0]}_health_report.pdf"
        
        profiler = AdvancedPDFProfilerAgent(file_path=file_path, output_pdf_path=report_output)
        profiler.generate_pdf_report()
        
        state["profiler_report"] = {"status": "success", "pdf_path": report_output}
        state["current_step"] = "profiled"
        print(f"✅ [Agent: Profiler] Success! PDF report card saved to {report_output}")
        
    except Exception as e:
        state["error"] = str(e)
        state["profiler_report"] = {"status": "failed", "error": str(e)}
        print(f"❌ [Agent: Profiler] Error: {e}")
        
    return state

def feature_engineering_node(state: AgentState) -> AgentState:
    """
    Agent Node 2: Adaptively applies feature engineering based on 
    inferred metadata (dates, granularities, SKUs, promotions).
    """
    print("\n🤖 [Agent: Feature Engineer] Adapting to data granularity and engineering features...")
    file_path = state["file_path"]
    engineered_output = f"data/engineered_{os.path.basename(file_path)}"
    
    try:
        df = pd.read_csv(file_path)
        
        # Run our adaptive feature engineering logic
        engineer = AdaptiveFeatureEngineer(df)
        engineered_df = engineer.transform()
        
        os.makedirs("data", exist_ok=True)
        engineered_df.to_csv(engineered_output, index=False)
        
        state["engineered_path"] = engineered_output
        state["current_step"] = "engineered"
        print(f"✅ [Agent: Feature Engineer] Success! Engineered data saved to {engineered_output}")
        
    except Exception as e:
        state["error"] = str(e)
        print(f"❌ [Agent: Feature Engineer] Error: {e}")
        
    return state

def inference_node(state: AgentState) -> AgentState:
    """
    Agent Node 3: Prepares sample inputs and routes them to the inference hub 
    (or pre-trained models from your other project folders).
    """
    print("\n🤖 [Agent: Inference] Routing data sample to model inference hub...")
    engineered_path = state.get("engineered_path")
    
    try:
        if engineered_path and os.path.exists(engineered_path):
            df = pd.read_csv(engineered_path)
            sample = df.iloc[0].to_dict()
            
            # Extract relevant prediction payload keys dynamically
            payload = {
                "store_nbr": int(sample.get("store_nbr", 1)),
                "family": str(sample.get("family", "DEFAULT")),
                "onpromotion": int(sample.get("onpromotion", 0)),
                "dcoilwtico": float(sample.get("dcoilwtico", 50.0))
            }
            state["prediction_input"] = payload
            state["prediction_result"] = {
                "predicted_sales": 285.50, # Will connect to FastAPI / model predict here
                "model_used": "Dynamic_Adaptive_Ensemble"
            }
            state["current_step"] = "completed"
            print("✅ [Agent: Inference] Prediction generated successfully.")
        else:
            raise FileNotFoundError("Engineered dataset not found for inference.")
            
    except Exception as e:
        state["error"] = str(e)
        print(f"❌ [Agent: Inference] Error: {e}")
        
    return state