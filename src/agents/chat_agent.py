import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from src.agents.workflow import build_supply_chain_workflow

class SupplyChainCopilotChat:
    def __init__(self):
        # 1. Initialize the Hugging Face LLM Brain
        # Ensure you have set your token: export HUGGINGFACEHUB_API_TOKEN="your_token"
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not api_token:
            print("⚠️ Warning: HUGGINGFACEHUB_API_TOKEN not found in environment variables. Running in offline/mock mode.")
            self.llm = None
        else:
            self.llm = HuggingFaceHub(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
            )
            
        # 2. Set up Conversation Memory for chat history
        self.memory = ConversationBufferMemory()
        if self.llm:
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=True
            )
        else:
            self.conversation = None

        # 3. Load our compiled LangGraph workflow
        self.workflow_app = build_supply_chain_workflow()

    def handle_user_prompt(self, prompt: str, data_path: str = "data/train.csv") -> dict:
        """
        Interprets the user's natural language request. If it involves running 
        the pipeline, it triggers the LangGraph workflow. Otherwise, it chats via LLM.
        """
        prompt_lower = prompt.lower()
        
        # Check if user prompt triggers the pipeline execution
        if any(keyword in prompt_lower for keyword in ["profile", "run pipeline", "predict", "analyze data", "features"]):
            print(f"🤖 [LLM Supervisor] Intent detected: Running autonomous supply chain workflow for '{data_path}'...")
            
            initial_state = {
                "file_path": data_path,
                "profiler_report": None,
                "engineered_path": None,
                "prediction_input": None,
                "prediction_result": None,
                "current_step": "initialized",
                "error": None
            }
            
            # Execute the LangGraph workflow we built earlier
            final_state = self.workflow_app.invoke(initial_state)
            
            return {
                "response_type": "workflow_execution",
                "message": "Successfully executed autonomous pipeline via LangGraph!",
                "details": final_state
            }
        else:
            # Standard conversational response using Hugging Face LLM
            if self.conversation:
                ai_reply = self.conversation.predict(input=prompt)
            else:
                ai_reply = "I am operating in offline mode. Ask me to 'profile my dataset' or 'run predictions' to trigger the pipeline!"
                
            return {
                "response_type": "chat",
                "message": ai_reply
            }

if __name__ == "__main__":
    # Test conversational agent interface
    copilot = SupplyChainCopilotChat()
    
    # Test natural language trigger
    result = copilot.handle_user_prompt("Can you profile my dataset and run predictions?")
    print("\n🏁 Chat Agent Response:")
    print(result)