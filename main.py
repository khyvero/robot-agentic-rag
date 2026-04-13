import argparse
from core.orchestrator import Orchestrator

if __name__ == "__main__":
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Robot Arm LLM Controller")
    parser.add_argument("--mode", type=str, default="zero_shot", 
                        choices=["zero_shot", "single_rag", "dual_rag"],
                        help="Operational mode: zero_shot, single_rag, or dual_rag")
    args = parser.parse_args()

    # Initialize Orchestrator
    orchestrator = Orchestrator()
    
    print("--- Robot Arm LLM Controller ---")
    print(f"Mode: {args.mode}")
    
    while True:
        try:
            user_input = input("\nEnter your command (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Plan the mission
            mission = orchestrator.plan_mission(user_input, mode=args.mode)
            
            # Execute the mission
            mission.execute()
            
        except Exception as e:
            print(f"Error: {e}")
