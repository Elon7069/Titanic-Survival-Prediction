import os
import time
import argparse

def run_step(script_name, description):
    """Run a specific step in the ML pipeline."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    exit_code = os.system(f"python src/{script_name}.py")
    end_time = time.time()
    
    if exit_code != 0:
        print(f"\nERROR: {script_name}.py exited with code {exit_code}")
        return False
    
    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    return True

def main():
    """Run the complete ML pipeline."""
    parser = argparse.ArgumentParser(description='Titanic Survival Prediction Pipeline')
    parser.add_argument('--visualize', action='store_true', help='Run data visualization')
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--train', action='store_true', help='Run model training')
    parser.add_argument('--predict', action='store_true', help='Run model prediction')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    
    args = parser.parse_args()
    
    # If no arguments provided, run the entire pipeline
    if not any([args.visualize, args.preprocess, args.train, args.predict, args.all]):
        args.all = True
    
    print("\n" + "="*80)
    print("TITANIC SURVIVAL PREDICTION PIPELINE")
    print("="*80 + "\n")
    
    pipeline_start = time.time()
    
    # Run data visualization
    if args.all or args.visualize:
        if not run_step("visualize", "DATA VISUALIZATION"):
            return
    
    # Run data preprocessing
    if args.all or args.preprocess:
        if not run_step("preprocess", "DATA PREPROCESSING"):
            return
    
    # Run model training
    if args.all or args.train:
        if not run_step("train", "MODEL TRAINING"):
            return
    
    # Run model prediction
    if args.all or args.predict:
        if not run_step("predict", "MODEL PREDICTION"):
            return
    
    pipeline_end = time.time()
    
    print("\n" + "="*80)
    print(f"PIPELINE COMPLETED in {pipeline_end - pipeline_start:.2f} seconds")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 