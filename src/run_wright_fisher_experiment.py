import numpy as np
import yaml
import json
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

from simulator import SpatialLineageSimulator
from kalman_filter import StandardKalmanFilter

# Ensure Seaborn styling matches your other scripts
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def calculate_time_resolved_metrics(true_freqs, estimated_freqs, epsilon=1e-10):
    """
    Calculates error metrics at every time step (T, L) -> (T,)
    """
    # Smooth zero probabilities to avoid log(0) and division by zero
    P = np.clip(true_freqs, epsilon, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True)
    
    Q = np.clip(estimated_freqs, epsilon, 1.0)
    Q = Q / np.sum(Q, axis=1, keepdims=True)
    
    metrics_over_time = {}
    
    # 1. RMSE (Root Mean Squared Error)
    metrics_over_time['RMSE'] = np.sqrt(np.mean((P - Q)**2, axis=1))
    
    # 2. TVD (Total Variation Distance)
    metrics_over_time['TVD'] = 0.5 * np.sum(np.abs(P - Q), axis=1)
    
    # 3. KL Divergence
    metrics_over_time['KL_Divergence'] = np.sum(P * np.log(P / Q), axis=1)
    
    # 4. JS Distance (Square root of JS Divergence for metric properties)
    # scipy's jensenshannon computes the distance directly for 1D arrays, so we loop over time
    js_dists = np.zeros(P.shape[0])
    for t in range(P.shape[0]):
        js_dists[t] = jensenshannon(P[t], Q[t])
    metrics_over_time['JS_Distance'] = js_dists
    
    return metrics_over_time

def main():
    print("--- Starting Wright-Fisher Perfect Control Experiment ---")
    
    # 1. Configuration for 1-Deme Control
    config = {
        "simulation_mode": "wright_fisher",
        "network_topology": "global",
        "output_path": "./data/temp_wf_control.npz",
        "seed": 42,
        "grid_size": 1,              # K = 1 deme
        "total_days": 200,
        "R0": 10,
        "generation_time": 1.0,
        "incubation_period": 2.5,
        "infectious_period": 6.5,
        "migration_rate": 0.0,       # No migration needed for 1 deme
        "overdispersion_k": 0.1,
        "total_lineages": 25,        # Keep it small so trajectories are readable
        "deme_census_pop": 5000,     # N = 5000 
        "initial_seeded_demes": 1,
        "initial_cases_per_seed": 1,
        "sequencing_capacity": 4990, # Perfect Observation Capacity
        "sampling_interval": 1       # Daily sampling
    }
    
    # 2. Run Simulation
    sim = SpatialLineageSimulator(config)
    infected = sim.run()
    observed = sim.sample_observations(infected)
    
    # Save temp file for the KF
    os.makedirs("./data", exist_ok=True)
    np.savez_compressed(config['output_path'], true_I=infected, obs_Y=observed, metadata=json.dumps(config))
    
    # 3. Run Kalman Filter
    kf = StandardKalmanFilter(config['output_path'])
    best_Ne, best_Ne_tilde, log_likelihood = kf.estimate_Ne()
    kf_trajectory = kf.get_filtered_trajectory(best_Ne_tilde)
    
    # Extract true frequencies from the KF's internal coarse-grained state
    true_counts_cg = kf.national_counts
    true_totals = np.sum(true_counts_cg, axis=1, keepdims=True)
    true_freqs = np.divide(true_counts_cg, true_totals, out=np.zeros_like(true_counts_cg, dtype=float), where=true_totals > 0)
    
    # Calculate time-resolved metrics
    metrics = calculate_time_resolved_metrics(true_freqs, kf_trajectory)
    
    # ---------------------------------------------------------
    # 4. PLOTTING
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Wright-Fisher Perfect Control: Simulation vs. Kalman Filter", fontsize=20, fontweight='bold')
    
    # --- Subplot 1: N vs Ne Bar Chart ---
    ax1 = plt.subplot(2, 2, 1)
    true_N = config['deme_census_pop']
    
    bars = ax1.bar(['True Census Pop ($N$)', 'Estimated Effective Pop ($N_e$)'], 
                   [true_N, best_Ne], 
                   color=['#3498db', '#2ecc71'])
    
    # Add exact values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (true_N * 0.02), 
                 f'{int(yval)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    ax1.set_title("Population Alignment", fontsize=14)
    ax1.set_ylabel("Population Size")
    
    # --- Subplot 3: Lineage Trajectories ---
    ax2 = plt.subplot(2, 1, 2)
    time_steps = np.arange(config['total_days'])
    
    colors = sns.color_palette("husl", kf_trajectory.shape[1])
    
    for l in range(kf_trajectory.shape[1]):
        # Plot True Frequencies (Solid lines)
        ax2.plot(time_steps, true_freqs[:, l], label=f'Lin {l} (True)' if l < 5 else "", 
                 color=colors[l], linewidth=2.5, alpha=0.8)
        
        # Plot KF Estimated Frequencies (Dashed lines)
        ax2.plot(time_steps, kf_trajectory[:, l], linestyle='--', color='black', alpha=0.6, linewidth=1.5)
    
    # Dummy lines for legend to explain styling
    ax2.plot([], [], color='gray', linewidth=2.5, label='True Freq')
    ax2.plot([], [], color='black', linestyle='--', linewidth=1.5, label='KF Est Freq')
    
    ax2.set_title("Lineage Trajectories: True vs. Kalman Filter", fontsize=14)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Frequency")
    ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    
# --- Subplot 2: Metrics Over Time ---
    ax3 = plt.subplot(2, 2, 2)
    metric_colors = ['#e74c3c', '#9b59b6', '#34495e', '#f39c12']
    
    for i, (metric_name, metric_data) in enumerate(metrics.items()):
        ax3.plot(time_steps, metric_data, label=metric_name, color=metric_colors[i], linewidth=2)
        
    ax3.set_title("Tracking Filtering Error Over Time", fontsize=14)
    ax3.set_xlabel("Days")
    ax3.set_ylabel("Error / Distance")
    ax3.set_yscale('log') # Log scale helps differentiate small differences near 0
    
    # ---> KEY ADDITION: Force the y-axis bounds to show the meaningful variance
    ax3.set_ylim([1e-4, 1e-0]) 
    
    ax3.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png = "./data/wright_fisher_validation.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    
    print(f"Success! Validation chart saved to: {output_png}")
    plt.show()

    # Cleanup temp file
    if os.path.exists(config['output_path']):
        os.remove(config['output_path'])

if __name__ == "__main__":
    main()