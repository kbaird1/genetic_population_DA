import numpy as np
import yaml
import json
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import copy

# Import your classes
from simulator import SpatialLineageSimulator
from kalman_filter import StandardKalmanFilter

def calculate_metrics(true_freqs, filtered_freqs, epsilon=1e-9):
    """
    Calculates both the daily error metrics and the overall average error 
    metrics across the entire simulation.
    """
    T = true_freqs.shape[0]
    
    # Initialize daily error arrays with NaN (so empty days don't skew plots)
    daily_rmse = np.full(T, np.nan)
    daily_tvd = np.full(T, np.nan)
    daily_js = np.full(T, np.nan)
    daily_kl = np.full(T, np.nan)
    
    for t in range(T):
        p = true_freqs[t]
        q = filtered_freqs[t]
        
        # Skip days with no data
        if np.sum(p) == 0 or np.sum(q) == 0:
            continue
            
        # Add epsilon to prevent divide-by-zero or log(0)
        p_clip = np.clip(p, epsilon, 1.0)
        q_clip = np.clip(q, epsilon, 1.0)
        
        # Re-normalize to perfect 1.0 distributions
        p_clip /= np.sum(p_clip)
        q_clip /= np.sum(q_clip)
        
        daily_rmse[t] = np.sqrt(np.mean((p_clip - q_clip)**2))
        daily_tvd[t] = 0.5 * np.sum(np.abs(p_clip - q_clip))
        daily_js[t] = jensenshannon(p_clip, q_clip)
        daily_kl[t] = entropy(p_clip, q_clip)
        
    # Calculate the averages ignoring the NaN days
    avg_rmse = float(np.nanmean(daily_rmse)) if not np.all(np.isnan(daily_rmse)) else 0.0
    avg_tvd = float(np.nanmean(daily_tvd)) if not np.all(np.isnan(daily_tvd)) else 0.0
    avg_js = float(np.nanmean(daily_js)) if not np.all(np.isnan(daily_js)) else 0.0
    avg_kl = float(np.nanmean(daily_kl)) if not np.all(np.isnan(daily_kl)) else 0.0

    # Convert NumPy arrays to standard Python lists for JSON, turning NaN into None (null in JSON)
    def clean_list(arr):
        return [None if np.isnan(x) else float(x) for x in arr]

    return {
        "averages": {
            "RMSE": avg_rmse,
            "TVD": avg_tvd,
            "JS_Distance": avg_js,
            "KL_Divergence": avg_kl
        },
        "daily": {
            "RMSE": clean_list(daily_rmse),
            "TVD": clean_list(daily_tvd),
            "JS_Distance": clean_list(daily_js),
            "KL_Divergence": clean_list(daily_kl)
        }
    }

def run_suite():
    base_config_path = "./config/simulation_params.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Define our experimental suite
    experiments = {
        "0_Perfect_Panmictic_Control_Noiseless": {
            "simulation_mode": "original_seir",
            "network_topology": "global",
            "grid_size": 1,              
            "deme_census_pop": 500000,   
            "initial_seeded_demes": 50,
            "migration_rate": 0.0,       
            "sequencing_capacity": 500000 
        },
        "1_Panmictic_Deme_Noiseless": {
            "simulation_mode": "original_seir",
            "network_topology": "global",
            "grid_size": 100,            
            "deme_census_pop": 50,       
            "initial_seeded_demes": 5000, 
            "migration_rate": 5.0,       
            "sequencing_capacity": 1000000 
        },
        "2_Spatial_Global_Noiseless": {
            "simulation_mode": "original_seir",
            "network_topology": "global",
            "grid_size": 100, 
            "deme_census_pop": 100,
            "sequencing_capacity": 1000000, 
            "migration_rate": 0.1
        },
        "3_Spatial_Lattice_Noiseless": {
            "simulation_mode": "original_seir",
            "network_topology": "lattice",
            "grid_size": 100, 
            "deme_census_pop": 100,
            "sequencing_capacity": 1000000, 
            "migration_rate": 0.1
        },
        "4_Spatial_Lattice_Noisy": {
            "simulation_mode": "original_seir",
            "network_topology": "lattice",
            "grid_size": 100, 
            "deme_census_pop": 100,
            "sequencing_capacity": 30, 
            "migration_rate": 0.1
        },
        "5_Superspreader_Jackpot_Noiseless": {
            "simulation_mode": "jackpot_negbinom",
            "network_topology": "global",
            "grid_size": 100,
            "deme_census_pop": 100,
            "sequencing_capacity": 1000000,
            "migration_rate": 0.1,
            "overdispersion_k": 0.05 
        },
        "6_Superspreader_Jackpot_Noisy": {
            "simulation_mode": "jackpot_negbinom",
            "network_topology": "global",
            "grid_size": 100,
            "deme_census_pop": 100,
            "sequencing_capacity": 30,
            "migration_rate": 0.1,
            "overdispersion_k": 0.05 
        }
    }

    results = {}
    trajectories = {}

    os.makedirs("./data/experiments", exist_ok=True)

    for exp_name, modifications in experiments.items():
        print(f"\n{'='*50}\nRunning Experiment: {exp_name}\n{'='*50}")
        
        # Modify config
        config = copy.deepcopy(base_config)
        for k, v in modifications.items():
            config[k] = v
        
        config['output_path'] = f"./data/experiments/{exp_name}.npz"
        
        # 1. Run Simulator
        sim = SpatialLineageSimulator(config)
        infected_data = sim.run()
        observed_data = sim.sample_observations(infected_data)
        
        np.savez_compressed(
            config['output_path'], 
            true_I=infected_data, 
            obs_Y=observed_data,
            metadata=json.dumps(config)
        )
        
        # 2. Run Kalman Filter
        kf = StandardKalmanFilter(config['output_path'])
        
        # Reconstruct True National Frequencies for metric comparison
        true_national_I = np.sum(kf.true_I, axis=1) 
        cg_true_counts = kf._coarse_grain(true_national_I, min_freq_threshold=0.01)
        true_totals = np.sum(cg_true_counts, axis=1, keepdims=True)
        true_freqs = np.divide(cg_true_counts, true_totals, out=np.zeros_like(cg_true_counts, dtype=float), where=true_totals>0)

        # Estimate Ne
        best_Ne, best_Ne_tilde, max_ll = kf.estimate_Ne()
        
        # Get Trajectory
        kf_trajectory = kf.get_filtered_trajectory(best_Ne_tilde)
        
        # 3. Calculate Errors (Now returns both averages and daily lists)
        metrics = calculate_metrics(true_freqs, kf_trajectory)

        # Store Results (Averages go to summary_results)
        results[exp_name] = {
            "True_Peak_N": int(kf.peak_true_infections),
            "Inferred_Ne": float(best_Ne),
            "Max_LogLikelihood": float(max_ll),
            **metrics["averages"]  # Unpacks RMSE, JS, TVD, and KL averages
        }
        
        # Store Trajectories (Daily error arrays go to trajectories along with the freqs)
        trajectories[exp_name] = {
            "True_Freqs": true_freqs.tolist(),
            "KF_Freqs": kf_trajectory.tolist(),
            "Daily_Errors": metrics["daily"] # Clean lists ready for JSON and plotting
        }

    # Save outputs
    with open("./data/experiments/summary_results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open("./data/experiments/trajectories.json", "w") as f:
        json.dump(trajectories, f)
        
    print("\nAll experiments completed and saved to ./data/experiments/")

if __name__ == "__main__":
    run_suite()