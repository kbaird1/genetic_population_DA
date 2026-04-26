import numpy as np
import yaml
import json
import os
import copy
from simulator import SpatialLineageSimulator
from kalman_filter import StandardKalmanFilter
from run_experiments import calculate_metrics

def run_sweep():
    base_config_path = "./config/simulation_params.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    scenarios = {
        "Control_Panmictic": {
            "grid_size": 1, "deme_census_pop": 500000, "network_topology": "global", "migration_rate": 0.0
        },
        "Spatial_Global": {
            "grid_size": 100, "deme_census_pop": 100, "network_topology": "global", "migration_rate": 0.1
        },
        "Spatial_Lattice": {
            "grid_size": 100, "deme_census_pop": 100, "network_topology": "lattice", "migration_rate": 0.1
        }
    }

    sweep_results = {}

    for sc_name, mods in scenarios.items():
        print(f"\n--- Scenario: {sc_name} ---")
        sweep_results[sc_name] = []
        
        # 1. First, find the "Peak National Infection" to set the sweep ceiling
        config_dry = copy.deepcopy(base_config)
        for k, v in mods.items(): config_dry[k] = v
        sim_dry = SpatialLineageSimulator(config_dry)
        infected_dry = sim_dry.run()
        
        # Peak total infections across the whole country
        peak_infected = int(np.max(np.sum(infected_dry, axis=1)))
        print(f"  Detected Peak Infections: {peak_infected}")
        
        # 2. Generate 20 points geometrically from 1 to Peak (e.g., 1, 2, 5, 12, 30...)
        ct_values = np.geomspace(1, peak_infected, 20).astype(int)
        # Ensure we have unique values if the range is small
        ct_values = np.unique(ct_values)
        
        for ct in ct_values:
            print(f"  Testing c_t = {ct}...", end="\r")
            
            # Setup Config for actual run
            config = copy.deepcopy(base_config)
            for k, v in mods.items(): config[k] = v
            config['sequencing_capacity'] = int(ct)
            temp_path = f"./data/temp_sweep_{sc_name}.npz"
            config['output_path'] = temp_path
            
            # Run simulation with the specific capacity
            sim = SpatialLineageSimulator(config)
            infected = sim.run()
            observed = sim.sample_observations(infected)
            np.savez_compressed(temp_path, true_I=infected, obs_Y=observed, metadata=json.dumps(config))
            
            # Run KF
            kf = StandardKalmanFilter(temp_path)

            best_Ne, best_Ne_tilde, _ = kf.estimate_Ne()
            kf_trajectory = kf.get_filtered_trajectory(best_Ne_tilde)

            # FIX: Use the KF's own internal coarse-grained counts to build true_freqs
            # This ensures that the 'Abundant' + 'Other' columns match kf_trajectory exactly
            true_counts_cg = kf.national_counts # This is already coarse-grained in kf.__init__
            true_totals = np.sum(true_counts_cg, axis=1, keepdims=True)

            # Calculate true frequencies safely
            true_freqs = np.divide(
                true_counts_cg, 
                true_totals, 
                out=np.zeros_like(true_counts_cg, dtype=float), 
                where=true_totals > 0
            )

            # Now shapes are guaranteed to match (T, L_coarse)
            m = calculate_metrics(true_freqs, kf_trajectory)
            
            sweep_results[sc_name].append({
                "c_t": int(ct),
                "metrics": m["averages"]
            })
            
            if os.path.exists(temp_path): os.remove(temp_path)

    with open("./data/experiments/resolution_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=4)
    print("\nSweep Complete.")

if __name__ == "__main__":
    run_sweep()