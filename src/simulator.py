import numpy as np
import yaml

class SpatialLineageSimulator:
    def __init__(self, config_payload):
        # Check if we were passed a path (string) or a dictionary
        if isinstance(config_payload, str):
            with open(config_payload, 'r') as f:
                self.p = yaml.safe_load(f)
        elif isinstance(config_payload, dict):
            self.p = config_payload
        else:
            raise TypeError("config_payload must be a file path (str) or a dictionary")
            
        if 'seed' in self.p and self.p['seed'] is not None:
            np.random.seed(self.p['seed'])
        
        self.K = self.p['grid_size']**2
        self.L = self.p['total_lineages']
        self.T = self.p['total_days']
        self.mode = self.p['simulation_mode']
        
        self.I_history = np.zeros((self.T, self.K, self.L), dtype=np.uint32)
        
    def _run_wright_fisher_mode(self):
        """
        Synchronous time-stepped Wright-Fisher model. 
        Replaces the event queue for 'wright_fisher' mode to allow lineages to compete.
        """
        print(f"Running Synchronous Wright-Fisher Simulation (K={self.K})...")
        
        # Ne acts as our deme census population
        N = self.p.get('deme_census_pop', 50)
        m = self.p.get('migration_rate', 0.1) 
        
        initial_seeds = self.p.get('initial_seeded_demes', 100)
        uninfected_demes = list(range(self.K))
        np.random.shuffle(uninfected_demes)
        
        # 1. Initialization (Day 0)
        if self.K == 1:
            # Perfectly random starting allele frequencies
            probs = np.random.dirichlet(np.ones(self.L))
            self.I_history[0, 0, :] = np.random.multinomial(N, probs)
        else:
            for _ in range(initial_seeds):
                if not uninfected_demes: break
                deme_id = uninfected_demes.pop()
                probs = np.random.dirichlet(np.ones(self.L))
                self.I_history[0, deme_id, :] = np.random.multinomial(N, probs)
                
        # 2. Time-stepped propagation
        for t in range(1, self.T):
            # Find demes that were active in the previous generation
            active_demes_mask = np.sum(self.I_history[t-1], axis=1) > 0
            
            if not np.any(active_demes_mask):
                break # The virus went extinct globally
                
            # Calculate global allele frequencies for migration
            global_pool = np.sum(self.I_history[t-1], axis=0)
            global_freqs = global_pool / np.sum(global_pool)
            
            for k in range(self.K):
                current_pop = self.I_history[t-1, k]
                total_I = np.sum(current_pop)
                
                if total_I > 0:
                    # Deme is infected: Execute Wright-Fisher Drift + Migration
                    local_freqs = current_pop / total_I
                    expected_freqs = (1 - m) * local_freqs + m * global_freqs
                    
                    # Multinomial sampling determines the next generation's frequencies
                    self.I_history[t, k] = np.random.multinomial(N, expected_freqs)
                    
                elif np.random.rand() < m and total_I == 0:
                    # Uninfected demes have a probability of being seeded via migration
                    self.I_history[t, k] = np.random.multinomial(N, global_freqs)
                    
        print(f"Simulation finished. Synchronous WF propagated over {self.T} days.")
        return self.I_history
    
    def _precompute_seir_curve(self):
        """
        Calculates the deterministic SEIR curve for a single isolated deme.
        Used for 'original_seir' mode to replicate Yu et al. (2024).
        """
        N = self.p.get('deme_census_pop', 50) 
        R0 = self.p.get('R0', 10.0)
        sigma = 1.0 / self.p.get('incubation_period', 2.5)
        gamma = 1.0 / self.p.get('infectious_period', 6.5)
        beta = R0 * gamma
        
        S, E, I, R = float(N - 1), 0.0, 1.0, 0.0 
        
        I_curve = []
        
        while (I > 0.01 or E > 0.01) and len(I_curve) < self.T * 2:
            I_curve.append(I)
            
            force_of_infection = beta * (S / N) * I
            new_exposed = min(force_of_infection, S)
            new_infectious = sigma * E
            new_recovered = gamma * I
            
            S -= new_exposed
            E += (new_exposed - new_infectious)
            I += (new_infectious - new_recovered)
            R += new_recovered
            
        return np.array(I_curve, dtype=np.float32)

    def _generate_jackpot_curve(self):
        """
        Calculates a stochastic SIR branching process on-the-fly.
        Used for 'jackpot_negbinom' mode to simulate heavy-tailed outbreaks.
        """
        N = self.p.get('deme_census_pop', 50)
        R0 = self.p.get('R0', 10.0)
        gamma = 1.0 / self.p.get('infectious_period', 6.5)
        k_disp = self.p.get('overdispersion_k', 0.1)
        
        S, I = N - 1, 1
        I_curve = []
        
        while I > 0 and len(I_curve) < self.T * 2:
            I_curve.append(I)
            
            # Expected new infections
            mu = R0 * (S / N) * I * gamma
            
            if mu > 0:
                p_prob = k_disp / (mu + k_disp)
                # Overdispersed negative binomial offspring
                new_I = np.random.negative_binomial(k_disp, p_prob)
            else:
                new_I = 0
                
            new_I = min(new_I, S)
            
            # Recoveries are binomially distributed
            recoveries = np.random.binomial(I, gamma)
            
            S -= new_I
            I += new_I - recoveries
            
        return np.array(I_curve, dtype=np.float32)

    def run(self):
        """
        Replicates the exact deme simulation structure from Yu et al. (2024),
        with support for heavy-tailed superspreader events.
        """

        # --- NEW ROUTING LOGIC ---
        if self.mode == 'wright_fisher':
            print('running wright_fisher simulation')
            return self._run_wright_fisher_mode()
        # -------------------------
            
        print(f"Running Discrete Event Deme Simulation (K={self.K})...")
        
        uninfected_demes = list(range(self.K))
        np.random.shuffle(uninfected_demes)
        
        # ... (The rest of your original run method stays exactly the same)
        print(f"Running Discrete Event Deme Simulation (K={self.K})...")
        
        uninfected_demes = list(range(self.K))
        np.random.shuffle(uninfected_demes)
        
        if self.mode == 'original_seir':
            local_I_curve = self._precompute_seir_curve()
            
        event_queue = []
                
        # Initialize seeds based on YAML config
        initial_seeds = self.p.get('initial_seeded_demes', 1000)
        
        if self.K == 1:
            # PERFECT CONTROL: Seed the single deme (deme_id 0) multiple times
            for _ in range(initial_seeds):
                lineage_id = np.random.randint(0, self.L)
                event_queue.append((0, lineage_id, 0))
        else:
            # ORIGINAL BEHAVIOR: 1 seed per spatial deme
            for _ in range(initial_seeds):
                if not uninfected_demes: break
                deme_id = uninfected_demes.pop()
                lineage_id = np.random.randint(0, self.L)
                event_queue.append((deme_id, lineage_id, 0))
            
        events_processed = 0
        migration_rate = self.p.get('migration_rate', 1.0)
        k_disp = self.p.get('overdispersion_k', 0.1)
        
        while event_queue:
            current_deme, lineage_id, start_day = event_queue.pop(0)
            events_processed += 1
            
            # Fetch the outbreak trajectory based on mode
            if self.mode == 'original_seir':
                curve = local_I_curve
                # Standard Poisson transmission to new demes
                num_secondary_demes = np.random.poisson(migration_rate)
            else:
                curve = self._generate_jackpot_curve()
                # Superspreader transmission to new demes (Negative Binomial)
                p_nb = k_disp / (migration_rate + k_disp)
                num_secondary_demes = np.random.negative_binomial(k_disp, p_nb)
            
            end_day = min(start_day + len(curve), self.T)
            duration = end_day - start_day
            
            # Paste the curve into the massive history tensor
            if duration > 0:
                # Use += instead of = so multiple seeds in the same deme stack!
                # We cast to the same dtype to prevent NumPy casting errors
                self.I_history[start_day:end_day, current_deme, lineage_id] += curve[:duration].astype(self.I_history.dtype)
            
            # Schedule Secondary Infections
            if num_secondary_demes > 0 and uninfected_demes and duration > 0:
                weights = curve[:duration]
                if np.sum(weights) > 0:
                    probs = weights / np.sum(weights)
                    
                    transmission_delays = np.random.choice(
                        np.arange(duration), 
                        size=num_secondary_demes, 
                        p=probs
                    )
                    
                    for delay in transmission_delays:
                        if not uninfected_demes: break
                            
                        transmission_day = start_day + delay
                        
                        if transmission_day < self.T:
                            target_deme = uninfected_demes.pop()
                            event_queue.append((target_deme, lineage_id, transmission_day))
                            
        print(f"Simulation finished. {events_processed} localized outbreaks occurred.")
        return self.I_history

    def sample_observations(self, infected_tensor):
        """
        Applies the genomic sequencing capacity bottleneck and aggregates 
        data over the specified sampling interval (e.g., weekly).
        """
        obs = np.zeros_like(infected_tensor, dtype=np.uint32)
        cap = self.p.get('sequencing_capacity', 30)
        interval = self.p.get('sampling_interval', 1)
        
        # Iterate through the timeline in blocks of 'interval' days
        for t in range(0, self.T, interval):
            end_t = min(t + interval, self.T)
            
            # Aggregate all active infections across the week
            interval_infections = np.sum(infected_tensor[t:end_t], axis=0)
            
            for k in range(self.K):
                total_I = np.sum(interval_infections[k])
                
                if total_I > 0:
                    actual_sequences = int(min(total_I, cap))
                    
                    # We record the observation on the final day of the interval
                    record_day = end_t - 1 
                    
                    # if actual_sequences == int(total_I):
                    #     # Perfect Observation (Noiseless)
                    #     obs[record_day, k] = interval_infections[k]
                    # else:
                    #     # Incomplete Observation (Multinomial Noise)
                    probs = interval_infections[k] / total_I
                    probs = probs.astype('float64')
                    probs /= np.sum(probs)
                    
                    sampled = np.random.multinomial(actual_sequences, probs)
                    obs[record_day, k] = sampled.astype(np.uint32)
                        
        return obs