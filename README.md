# Spatial Lineage Simulator: Modeling Spatial Misspecification and Genetic Drift

## Overview
This repository contains a stochastic, spatial metapopulation simulator designed to generate ground-truth epidemic data. Its primary purpose is to benchmark Data Assimilation techniques (specifically a standard Kalman Filter) against complex, non-panmictic epidemiological scenarios. 

This project builds directly upon the foundational simulations of **Yu et al. (2024)**, which demonstrated that spatial clustering mimics extreme genetic drift. Our framework extends their methodology to prove that standard 1D Wright-Fisher models are inherently misspecified for real-world pathogens. Because these models lack mechanisms for spatial waves, migration bottlenecks, or heavy-tailed superspreading, they forcibly absorb all system variance into a single parameter: elevated genetic drift. This leads to a severe underestimation of the Effective Population Size ($N_e$). 

By rigorously demonstrating this misspecification, this framework provides the testing ground and motivation for developing next-generation joint-inference models capable of decoupling true genetic drift from spatial and behavioral transmission dynamics.

---

## Simulation Mechanics

The simulator operates as a dual-engine framework, highly modularized to test specific epidemiological breakdowns.

### 1. Intra-Deme Dynamics (Local Outbreaks)
* **Original SEIR Mode (`original_seir`):** Replicates the deterministic, continuous-time SEIR approximations used in Yu et al. Local outbreaks follow smooth, predictable epidemic curves.
* **Jackpot Mode (`jackpot_negbinom`):** Replaces the deterministic curve with a highly stochastic, discrete-time branching process parameterized by a Negative Binomial distribution. This simulates heavy-tailed, localized "superspreading" events.
* * **Wright-Fisher Mode (`wright_fisher`):** Serves as the mathematical control for the framework. It bypasses the continuous-time SEIR event queue entirely, simulating a synchronous, discrete-time Wright-Fisher process. In this mode, the generation time $\tau$ effectively becomes 1 time step, and the entire population of each deme is replaced synchronously. The mechanics rely on calculating the post-migration frequencies of each lineage, followed by a multinomial draw to determine the next generation's true genetic drift.
  
  The mathematical updates for a deme $k$ and lineage $l$ at time $t$ are governed by:
  
  1. **Pre-migration Frequency:** $$f_{t, k, l} = \frac{I_{t, k, l}}{N_k}$$
  
  2. **Post-migration Probability:** $$p_{t, k, l} = (1 - m) f_{t, k, l} + m \sum_{j} W_{k, j} f_{t, j, l}$$
  
  3. **Stochastic Next Generation:** $$\mathbf{I}_{t+1, k} \sim \text{Multinomial}(N_k, \mathbf{p}_{t, k})$$
  
  Where $N_k$ is the constant `deme_census_pop`, $m$ is the `migration_rate`, $W_{k,j}$ is the spatial network weight matrix (global or lattice), and $\mathbf{p}_{t, k}$ is the vector of all lineage probabilities in deme $k$.

### 2. Inter-Deme Transmission (Spatial Topology)
* **Global Network (`global`):** Random mixing. A source deme is equally likely to seed any other deme in the network, perfectly matching standard metapopulation assumptions.
* **Lattice Network (`lattice`):** Spatial wave-like spread. Transmission is restricted to immediate geometric neighbors on a 2D grid, creating intense spatial autocorrelation and geographic bottlenecks.

---

## The Observation Model

Unlike foundational models that assume perfect, noiseless observation of lineage frequencies, this framework explicitly models the genomic surveillance system. 
* **Sequencing Capacity ($C_{seq}$):** Introduces a strict daily/weekly bottleneck on the maximum number of viral genomes a lab can process per deme. Outbreaks exceeding this limit are subjected to Multinomial sampling noise.
* **Temporal Granularity:** Supports customizable sampling intervals, allowing the simulation to aggregate surveillance data daily, weekly, or bi-weekly before assimilation.

---

## Data Assimilation (The Kalman Filter)

The assimilation engine uses a Standard Kalman Filter designed to enforce the panmictic assumption (aggregating all spatial data nationally). 
* **Process Model:** Pathogen evolution is modeled via neutral genetic drift. To maintain analytical tractability, frequencies undergo a Fisher-Tukey variance-stabilizing transformation (square root), making the process noise a constant scalar defined entirely by $N_e$.
* **Maximum Likelihood:** The filter sweeps over a logarithmic grid of candidate values, identifying the optimal daily variance parameter ($\tilde{N}_e$) that maximizes the cumulative log-likelihood of the innovations. 

The core output is the comparison between the true simulated peak census infections ($N$) and the inferred $N_e$, quantifying the exact magnitude of the model's misspecification.

---

## Configuration

All experiments are controlled via `config/simulation_params.yaml`. Key toggles include:
* **Master Toggles:** `simulation_mode` and `network_topology`.
* **Spatial Scale:** Grid size (total demes) and total simulation days.
* **Biological Parameters:** $R_0$, generation time, incubation, and infectious periods.
* **Stochastic Parameters:** Migration rates and Negative Binomial overdispersion ($k_{disp}$).
* **Observation Settings:** Sequencing capacity bottlenecks and sampling intervals.

---

## Usage

1. Adjust the hyperparameters in `config/simulation_params.yaml` to configure your experimental scenario.
2. Execute the simulator to generate the ground-truth and observational data:
```bash
python run_simulation.py
```
1. The terminal will output summary statistics, including the peak active infections, cumulative infections, and the total number of genomic sequences successfully sampled.
2. The resulting `sim_output.npz` file will be saved to your specified `output_path`, ready for Phase II (Kalman Filter benchmarking).