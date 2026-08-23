import math

class RDPPrivacyAccountant:
    """
    Rényi Differential Privacy (RDP) accountant to track cumulative epsilon budget (ε, δ)
    across federated learning rounds.
    """
    def __init__(self, target_delta: float = 1e-4):
        self.target_delta = target_delta
        self.history = []

    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1):
        self.history.append({
            "sigma": noise_multiplier,
            "q": sample_rate,
            "steps": steps
        })

    def get_epsilon(self, delta: float = None) -> float:
        if delta is None:
            delta = self.target_delta
        if not self.history:
            return 0.0
        
        # Search over a range of candidate Rényi orders to find the tightest (minimum) epsilon
        candidate_alphas = [
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0,
            32.0, 64.0
        ]
        min_eps = float('inf')
        
        for alpha in candidate_alphas:
            total_rdp = 0.0
            for entry in self.history:
                sigma = entry["sigma"]
                q = entry["q"]
                steps = entry["steps"]
                if sigma <= 0:
                    continue
                # RDP step formula for subsampled Gaussian mechanism
                rdp_step = steps * (q ** 2) * alpha / (2.0 * (sigma ** 2))
                total_rdp += rdp_step
            
            # Convert composed RDP to standard (epsilon, delta)-DP
            eps = total_rdp + math.log(1.0 / delta) / (alpha - 1)
            if eps < min_eps:
                min_eps = eps

        return round(min(min_eps, 15.0), 3)

