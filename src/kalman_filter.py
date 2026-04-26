import numpy as np
import json
import os

class StandardKalmanFilter:
    def __init__(self, data_path):
        # 1. Load the Simulated Data
        data = np.load(data_path)
        self.obs_Y = data['obs_Y']  # Shape: (T, K, L)
        self.metadata = json.loads(str(data['metadata']))
        
        self.T, self.K, self.L = self.obs_Y.shape
        
        # 2. Force the Panmictic Assumption (Aggregate Demes)
        self.national_counts = np.sum(self.obs_Y, axis=1) # Shape: (T, L)
        
        # Apply Coarse Graining
        self.national_counts = self._coarse_grain(self.national_counts, min_freq_threshold=0.01)
        self.L = self.national_counts.shape[1] 
        
        self.total_sampled_per_day = np.sum(self.national_counts, axis=1) # Shape: (T,)
        
        # 3. True Population Metric (for comparison later)
        self.true_I = data['true_I']
        self.true_national_I = np.sum(self.true_I, axis=(1, 2))
        self.peak_true_infections = int(np.max(self.true_national_I))
        
    def _coarse_grain(self, counts, min_freq_threshold=0.01):
        total_daily = np.sum(counts, axis=1, keepdims=True)
        safe_total = np.where(total_daily > 0, total_daily, 1)
        freqs = counts / safe_total
        
        max_freqs = np.max(freqs, axis=0)
        # --- UPDATE: Save the mask and the count of kept lineages ---
        self.keep_mask = max_freqs >= min_freq_threshold
        self.num_abundant = np.sum(self.keep_mask)
        
        if np.all(self.keep_mask):
            return counts 
            
        abundant_counts = counts[:, self.keep_mask]
        rare_counts = counts[:, ~self.keep_mask]
        other_counts = np.sum(rare_counts, axis=1, keepdims=True)
        
        coarse_counts = np.hstack([abundant_counts, other_counts])
        return coarse_counts
    
    # def run_filter(self, Ne_tilde):
    #     """
    #     Runs the variance-stabilized 1D Kalman Filter exactly as described 
    #     in Yu et al. (2024). Note that the input is Ne_tilde (the daily scaled Ne).
    #     """
    #     total_log_likelihood = 0.0
        
    #     # 1. Prepare Observations (Convert to frequencies)
    #     z_raw = np.zeros_like(self.national_counts, dtype=np.float64)
    #     valid_days = self.total_sampled_per_day > 0
    #     z_raw[valid_days] = self.national_counts[valid_days] / self.total_sampled_per_day[valid_days, None]
        
    #     # Apply Fisher-Tukey (Square Root) Transformation
    #     z_sqrt = np.sqrt(z_raw)
        
    #     # 2. Initialize State Arrays
    #     if self.total_sampled_per_day[0] > 0:
    #         phi_hat = z_sqrt[0, :].copy()
    #     else:
    #         phi_hat = np.sqrt(np.ones(self.L) / self.L)
            
    #     P = np.ones(self.L) * 1e-4
        
    #     # ---> KEY CHANGE: Process noise uses Ne_tilde
    #     Q = 1.0 / (4.0 * Ne_tilde)
        
    #     for t in range(1, self.T):
    #         S_t = self.total_sampled_per_day[t]
            
    #         # --- PREDICT STEP ---
    #         phi_pred = phi_hat  
    #         P_pred = P + Q
            
    #         if S_t == 0:
    #             phi_hat = phi_pred
    #             P = P_pred
    #             continue
                
    #         # --- UPDATE STEP ---
    #         R_t = 1.0 / (4.0 * S_t)
    #         y_t = z_sqrt[t] - phi_pred
    #         S_cov = P_pred + R_t
    #         K_t = P_pred / S_cov
            
    #         phi_hat = phi_pred + K_t * y_t
    #         P = (1.0 - K_t) * P_pred
    #         phi_hat = np.clip(phi_hat, 0.0, 1.0)
            
    #         # Log-Likelihood
    #         log_likelihood_t = np.sum(-0.5 * np.log(2 * np.pi * S_cov) - (y_t ** 2) / (2 * S_cov))
    #         total_log_likelihood += log_likelihood_t
                
    #     return total_log_likelihood
    
    
    
    def run_filter(self, Ne_tilde):
        """
        Runs the variance-stabilized 1D Kalman Filter exactly as described 
        in Yu et al. (2024). Note that the input is Ne_tilde (the daily scaled Ne).
        """
        total_log_likelihood = 0.0
        
        # 1. Prepare Observations (Convert to frequencies)
        z_raw = np.zeros_like(self.national_counts, dtype=np.float64)
        valid_days = self.total_sampled_per_day > 0
        z_raw[valid_days] = self.national_counts[valid_days] / self.total_sampled_per_day[valid_days, None]
        
        # Apply Fisher-Tukey (Square Root) Transformation
        z_sqrt = np.sqrt(z_raw)
        
        # 2. Initialize State Arrays
        if self.total_sampled_per_day[0] > 0:
            phi_hat = z_sqrt[0, :].copy()
        else:
            phi_hat = np.sqrt(np.ones(self.L) / self.L)
            
        P = np.ones(self.L) * 1e-4
        
        # Process noise uses Ne_tilde
        Q = 1.0 / (4.0 * Ne_tilde)
        
        for t in range(1, self.T):
            S_t = self.total_sampled_per_day[t]
            
            # --- PREDICT STEP ---
            phi_pred = phi_hat  
            P_pred = P + Q
            
            if S_t == 0:
                phi_hat = phi_pred
                P = P_pred
                continue
                
            # --- UPDATE STEP ---
            R_t = 1.0 / (4.0 * S_t)
            y_t = z_sqrt[t] - phi_pred
            S_cov = P_pred + R_t
            K_t = P_pred / S_cov
            
            # ---> NEW ADDITION: Enforce Rare Lineage Assumption for Log-Likelihood
            rare_threshold = 0.05  # Only evaluate lineages under 10% frequency
            rare_mask = (phi_pred ** 2) < rare_threshold
            
            # Only sum the log-likelihood for valid rare lineages
            if np.any(rare_mask):
                log_likelihood_t = np.sum(-0.5 * np.log(2 * np.pi * S_cov[rare_mask]) - (y_t[rare_mask] ** 2) / (2 * S_cov[rare_mask]))
                total_log_likelihood += log_likelihood_t
            
            # Update state variables
            phi_hat = phi_pred + K_t * y_t
            P = (1.0 - K_t) * P_pred
            phi_hat = np.clip(phi_hat, 0.0, 1.0)
                
        return total_log_likelihood

    def estimate_Ne(self):
        """Searches for the Ne_tilde that maximizes Log-Likelihood, then converts to Ne"""
        print("Running Kalman Filter over Ne_tilde grid...")
        
        Ne_tilde_candidates = np.logspace(1, 4, 200)
        best_Ne_tilde = None
        max_ll = -np.inf
        
        for Ne_tilde in Ne_tilde_candidates:
            ll = self.run_filter(Ne_tilde)
            if ll > max_ll:
                max_ll = ll
                best_Ne_tilde = Ne_tilde
                
        # ---> KEY CHANGE: Convert daily variance (Ne_tilde) to generational variance (Ne)
        tau = self.metadata.get('generation_time', 5.0)
        best_Ne = best_Ne_tilde / tau
        
        return best_Ne, best_Ne_tilde, max_ll

    def get_filtered_trajectory(self, Ne):
        """
        Returns the smoothed frequency trajectory based on optimal Ne.
        """
        z_raw = np.zeros_like(self.national_counts, dtype=np.float64)
        valid_days = self.total_sampled_per_day > 0
        z_raw[valid_days] = self.national_counts[valid_days] / self.total_sampled_per_day[valid_days, None]
        
        z_sqrt = np.sqrt(z_raw)
        
        if self.total_sampled_per_day[0] > 0:
            phi_hat = z_sqrt[0, :].copy()
        else:
            phi_hat = np.sqrt(np.ones(self.L) / self.L)
            
        P = np.ones(self.L) * 1e-4
        
        phi_history = np.zeros_like(z_sqrt)
        phi_history[0] = phi_hat
        
        Q = 1.0 / (4.0 * Ne)
        
        for t in range(1, self.T):
            S_t = self.total_sampled_per_day[t]
            
            phi_pred = phi_hat
            P_pred = P + Q
            
            if S_t == 0:
                phi_hat = phi_pred
                P = P_pred
                phi_history[t] = phi_hat
                continue
                
            R_t = 1.0 / (4.0 * S_t)
            
            y_t = z_sqrt[t] - phi_pred
            S_cov = P_pred + R_t
            K_t = P_pred / S_cov
            
            phi_hat = phi_pred + K_t * y_t
            P = (1.0 - K_t) * P_pred
            phi_hat = np.clip(phi_hat, 0.0, 1.0)
            
            phi_history[t] = phi_hat
            
        # Inverse transform back to raw frequencies
        f_history = phi_history ** 2
        f_history = f_history / np.sum(f_history, axis=1, keepdims=True)
        
        return f_history

if __name__ == "__main__":
    data_path = "./data/raw/sim_output.npz"
    if os.path.exists(data_path):
        kf = StandardKalmanFilter(data_path)
        
        print("--- Standard Kalman Filter (Phase II) ---")
        print(f"Simulation Topology: {kf.metadata.get('network_topology', 'Unknown')}")
        print(f"True Peak Daily Infections (Census Pop): {kf.peak_true_infections}")
        print("-" * 40)
        
        estimated_Ne, estimated_Ne_tilde, log_likelihood = kf.estimate_Ne()
        
        print(f"Estimated Effective Population Size (Ne): {estimated_Ne:.2f}")
        print(f"Log-Likelihood: {log_likelihood:.2f}")
        
        ratio = kf.peak_true_infections / estimated_Ne
        print("-" * 40)
        print(f"Result: The Standard Kalman Filter underestimated the peak population by a factor of {ratio:.2f}x")
    else:
        print(f"Data file not found at {data_path}")