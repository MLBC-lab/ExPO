"""
Command-line interface for ExPO.
"""
import argparse
import sys
import os
from pathlib import Path


def train_main():
    """Entry point for expo-train command."""
    try:
        from expo.training.trainer import train_expo_from_config
        from expo.config import ExperimentConfig
        
        parser = argparse.ArgumentParser(description="Train ExPO model")
        parser.add_argument("--config", type=str, required=True,
                           help="Path to configuration JSON file")
        parser.add_argument("--use-full-trainer", action="store_true",
                           help="Use the complete training script instead of CLI")
        args = parser.parse_args()
        
        config = ExperimentConfig.from_json(args.config)
        
        if args.use_full_trainer:
            # Run the full training script
            import subprocess
            project_root = Path(__file__).parent.parent
            script_path = project_root / "scripts" / "train_expo.py"
            
            if script_path.exists():
                print("Running full training script...")
                result = subprocess.run([
                    sys.executable, str(script_path), 
                    "--config", args.config
                ], cwd=project_root)
                sys.exit(result.returncode)
            else:
                print("Full training script not found. Using simplified training.")
        
        result = train_expo_from_config(config)
        print(f"\nTraining result: {result}")
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure ExPO is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


def eval_main():
    """Entry point for expo-eval command."""
    try:
        # Add the project root to the path for scripts import
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from scripts.eval_expo import main
        main()
    except ImportError as e:
        print(f"Error importing evaluation script: {e}")
        print("Evaluation functionality not available in this distribution.")
        sys.exit(1)


def demo_main():
    """Entry point for expo-demo command."""
    try:
        # Add the project root to the path for scripts import
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from scripts.run_full_demo_pipeline import main
        main()
    except ImportError as e:
        print(f"Error importing demo script: {e}")
        print("Demo functionality not available. Please run:")
        print("  python scripts/run_full_demo_pipeline.py")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m expo.cli <command>")
        print("Commands: train, eval, demo")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from args
    
    if command == "train":
        train_main()
    elif command == "eval":
        eval_main()
    elif command == "demo":
        demo_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)