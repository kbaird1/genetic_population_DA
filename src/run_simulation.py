from src import SpatialLineageSimulator
import numpy as np
import os
import json

def main():
    config_path = "./config/simulation_params.yaml"
    
    # 1. Run Simulator
    sim = SpatialLineageSimulator(config_path)
    print(f"Running simulation in '{sim.mode}' mode...")
    infected_data = sim.run()
    
    # 2. Sample Observations
    print("Applying genomic surveillance observation model...")
    observed_data = sim.sample_observations(infected_data)
    
    # 3. Dynamic Output Path & Directory Creation
    output_file = sim.p.get('output_path', "./data/raw/sim_output.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 4. Save as compressed NumPy file with embedded config metadata
    np.savez_compressed(
        output_file, 
        true_I=infected_data, 
        obs_Y=observed_data,
        metadata=json.dumps(sim.p) # <--- This embeds the exact YAML readout!
    )
    
    # Print Summary Statistics
    total_cumulative_infections = int(np.sum(infected_data))
    peak_infections_day = int(np.max(np.sum(infected_data, axis=(1, 2))))
    total_sequences = int(np.sum(observed_data))
    
    print(f"\n--- Simulation Complete ---")
    print(f"Seed Used: {sim.p.get('seed', 'None')}")
    print(f"Data saved to: {output_file}")
    print(f"Tensor Shape: {observed_data.shape} (Days, Demes, Lineages)")
    print(f"Total Cumulative Infections: {total_cumulative_infections}")
    print(f"Peak Active Infections in one day: {peak_infections_day}")
    print(f"Total Sequences Sampled: {total_sequences}")

if __name__ == "__main__":
    main()