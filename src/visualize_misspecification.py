import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math

# Set publication style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5

def plot_gotcha_multiplier(results):
    """Plots the True N vs Inferred Ne and highlights the Order of Magnitude difference."""
    labels = [" ".join(k.split('_')[1:]) for k in results.keys()] 
    true_N = [v['True_Peak_N'] for v in results.values()]
    inferred_Ne = [v['Inferred_Ne'] for v in results.values()]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, true_N, width, label='True Peak Census (N)', color='#2c3e50')
    rects2 = ax.bar(x + width/2, inferred_Ne, width, label='Inferred Effective Pop ($N_e$)', color='#e74c3c')

    ax.set_yscale('log')

    for i in range(len(labels)):
        if inferred_Ne[i] > 0:
            oom_diff = np.log10(true_N[i] / inferred_Ne[i])
            max_height = max(true_N[i], inferred_Ne[i])
            
            ax.annotate(f"{oom_diff:.1f}",
                        xy=(x[i], max_height),
                        xytext=(0, 10), 
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontweight='bold', color='#c0392b', fontsize=12)

    ax.set_ylabel('Population Size (Log Scale)', fontweight='bold')
    ax.set_title('True Census vs. Inferred Effective Population Size ($N_e$)', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs('./data/experiments', exist_ok=True)
    plt.savefig('./data/experiments/fig_1_gotcha.png', dpi=300)
    plt.show()

def plot_errors_vs_time_per_experiment(trajectories, results):
    """Creates a subplot for each experiment, plotting all 4 daily error metrics over time, 
       with averages displayed on the right of each graph."""
    experiments = list(trajectories.keys())
    num_exps = len(experiments)
    
    # Dynamically size the grid based on how many experiments you ran
    cols = 2 if num_exps <= 4 else 3
    rows = math.ceil(num_exps / cols)
    
    # Increased width slightly to accommodate the legends on the right
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), sharex=True, sharey=True)
    
    # Ensure axes is always a 1D array for easy iteration, even if it's 1x1
    if num_exps == 1: 
        axes = [axes]
    else: 
        axes = axes.flatten()
    
    metrics_names = ['RMSE', 'TVD', 'JS_Distance', 'KL_Divergence']
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c'] 
    
    for i, exp_name in enumerate(experiments):
        ax = axes[i]
        
        daily_errors = trajectories[exp_name]['Daily_Errors']
        T = len(daily_errors['RMSE'])
        
        # Clean title
        clean_title = " ".join(exp_name.split('_')[1:])
        ax.set_title(clean_title, fontweight='bold', fontsize=12)
        
        # Plot the 4 lines for this specific simulation
        for j, metric in enumerate(metrics_names):
            clean_metric_name = metric.replace('_', ' ')
            
            # Fetch the overall average from the results dictionary
            avg_val = results[exp_name].get(metric, 0.0)
            
            # Create a label that includes the name AND the average formatted to 3 decimals
            label_text = f"{clean_metric_name}\n(Avg: {avg_val:.3f})"
            
            # Convert the list back to a numpy float array safely
            y_data = np.array(daily_errors[metric], dtype=float)
            
            ax.plot(range(T), y_data, label=label_text, color=colors[j], linewidth=2)
            
        ax.set_ylabel('Error')
        if i >= num_exps - cols: # Only add X labels to the bottom row
            ax.set_xlabel('Simulation Day')
            
        # Place the legend OUTSIDE the graph area, just to the right (center-right alignment)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, 
                  frameon=True, facecolor='white', edgecolor='lightgray')

    # Hide any unused subplots
    for j in range(num_exps, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Error Metrics Over Time By Simulation Scenario', fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout to make sure there's enough room on the right side for the legends
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.35) # Adds horizontal space between columns so legends don't overlap
    
    plt.savefig('./data/experiments/fig_2_time_vs_error_grid.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    try:
        with open("./data/experiments/summary_results.json", "r") as f:
            results = json.load(f)
        with open("./data/experiments/trajectories.json", "r") as f:
            trajectories = json.load(f)
    except FileNotFoundError:
        print("Data files not found. Please run run_experiments.py first.")
        exit(1)

    # 1. Bar Chart: True N vs Inferred Ne multiplier
    plot_gotcha_multiplier(results)
    
    # 2. Line Chart Grid: 4 Error metrics vs Time with Averages on the right
    plot_errors_vs_time_per_experiment(trajectories, results)