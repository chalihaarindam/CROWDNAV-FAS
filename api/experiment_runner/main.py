# main.py
from bayesian_optimization import BayesianOptimization, EpsilonGreedy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Configuration of router parameters
parameter_bounds = np.array([
    [0, 1],   # route_randomization
    [0, 1],   # exploration_percentage
    [0, 1],   # static_info_weight
    [0, 1],   # dynamic_info_weight
    [0, 1],   # exploration_weight
    [0, 10000], # data_freshness_threshold
    [0, 60]   # re-routing_frequency
])

def wait_for_crowdnav():
    """Wait for crowdnav to be ready"""
    import requests
    import time
    max_attempts = 30
    for _ in range(max_attempts):
        try:
            response = requests.get('http://localhost:8080/monitor')
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def run_experiment(experiment_id):
    # Initialize optimization strategy
    
    bo = EpsilonGreedy(parameter_bounds, epsilon=0.1)
   
    #bo = BayesianOptimization(parameter_bounds)
    
    # Run optimization
    best_params, best_value, history = bo.optimize(num_iterations=50)
    
    # Save results with experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"epsilon_exp{experiment_id}_{timestamp}"
    
    # Create results directories
    os.makedirs("/code/results/data", exist_ok=True)
    os.makedirs("/code/results/visualizations", exist_ok=True)
    
    # Save CSV results
    results_df = pd.DataFrame({
        'iteration': history['iterations'],
        'value': history['values'],
        'best_value': history['best_values'],
        'parameters': history['params']
    })
    results_df.to_csv(f'/code/results/data/{filename_base}.csv', index=False)
    
    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['iteration'], results_df['value'], 'b-', label='Current Value', alpha=0.5)
    plt.plot(results_df['iteration'], results_df['best_value'], 'r-', label='Best Value')
    plt.title(f'Experiment {experiment_id}: Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Performance Metric')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/code/results/visualizations/{filename_base}_plot.png', dpi=300)
    plt.close()
    
    return best_value, best_params

def main():
    if not wait_for_crowdnav():
        print("Crowdnav not responding")
        return
    
    num_experiments = 10
    
    
    # Run multiple experiments
    results = []
    for i in range(num_experiments):
        print(f"\nRunning experiment {i+1}/{num_experiments}")
        best_value, best_params = run_experiment(i+1)
        results.append({
            'experiment': i+1,
            'best_value': best_value,
            'best_params': best_params
        })
        
    # Save summary of all experiments
    summary_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df.to_csv(f'/code/results/data/summary_{timestamp}.csv', index=False)

if __name__ == "__main__":
    main()
