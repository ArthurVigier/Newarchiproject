import torch
import torch.nn as nn

class EntropyRouter(nn.Module):
    def __init__(self, predictor_model: nn.Module, num_experts: int = 3, mc_samples: int = 5):
        super().__init__()
        self.predictor = predictor_model
        self.num_experts = num_experts
        self.mc_samples = mc_samples

    def compute_expert_entropy(self, z, expert_idx):
        # We assume the predictor has a way to condition on expert_idx if needed,
        # or we have N predictors. For now, assuming a single stochastic predictor
        # that implicitly routes or we measure general uncertainty.
        # Based on context: "Routing par entropie : sélectionne l'expert dont les 
        # prédictions stochastiques ont la plus faible variance"
        
        # Ensure dropout is active
        self.predictor.train()
        
        # MC Dropout passes
        preds = []
        for _ in range(self.mc_samples):
            # In a full implementation, predictor might take (z, expert_idx)
            preds.append(self.predictor(z))
            
        preds = torch.stack(preds) # (mc_samples, batch, latent_dim)
        
        # Variance as a proxy for entropy/uncertainty
        entropy = preds.var(dim=0).mean(dim=-1) # (batch,)
        return entropy

    def forward(self, z):
        """
        Returns the routing indices based on lowest entropy.
        """
        # In this simplified version, if all experts use the same predictor, 
        # entropy is the same. To make it MoE, we need N predictors, one per expert.
        pass

class MultiPredictorEntropyRouter(nn.Module):
    def __init__(self, predictors: nn.ModuleList, mc_samples: int = 5):
        super().__init__()
        self.predictors = predictors
        self.num_experts = len(predictors)
        self.mc_samples = mc_samples

    def forward(self, z):
        batch_size = z.size(0)
        entropies = torch.zeros(batch_size, self.num_experts, device=z.device)
        
        for i, predictor in enumerate(self.predictors):
            predictor.train() # Enable MC Dropout
            with torch.no_grad(): # We don't backprop through routing logic itself here
                preds = torch.stack([predictor(z) for _ in range(self.mc_samples)])
                # Variance over MC samples, mean over latent dimensions
                entropy = preds.var(dim=0).mean(dim=-1) 
                entropies[:, i] = entropy
                
        # Select expert with the LOWEST entropy (least surprised)
        selected_experts = torch.argmin(entropies, dim=1)
        return selected_experts, entropies

class SurvivalExpert(nn.Module):
    def __init__(self, latent_dim=512, n_neurons=256, 
                 dormancy_threshold=30, death_threshold=50, rebirth_noise=0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_neurons = n_neurons
        self.dormancy_threshold = dormancy_threshold
        self.death_threshold = death_threshold
        self.rebirth_noise = rebirth_noise
        
        # Poids non-entraînés par backprop !
        self.weights = nn.Parameter(
            torch.randn(n_neurons, latent_dim) * 0.01,
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(n_neurons),
            requires_grad=False
        )
        self.output_weights = nn.Parameter(
            torch.randn(n_neurons, latent_dim) * 0.01,
            requires_grad=False
        )
        
        # État interne persistant (non-paramètres)
        self.register_buffer('survival_scores', torch.zeros(n_neurons))
        self.register_buffer('dormancy_counters', torch.zeros(n_neurons))
        
        # Historique d'activation pour la mise à jour darwinienne (batch temp storage)
        self._last_active_mask = None

    def forward(self, z):
        """
        z: (batch, latent_dim)
        """
        # Linear projection
        h = torch.matmul(z, self.weights.T) + self.bias # (batch, n_neurons)
        
        # Activation stochastique / binaire
        is_active_base = (h > 0)
        
        # Applique le masque de dormance
        # Un neurone dormant ne peut pas s'activer
        not_dormant = (self.dormancy_counters <= self.dormancy_threshold).unsqueeze(0)
        is_active = is_active_base & not_dormant
        
        # Sauvegarde pour la phase de reward (somme sur le batch pour simplifier 
        # l'attribution locale, ou on peut garder par sample)
        self._last_active_mask = is_active.float().mean(dim=0) > 0.1 # Actif si utile pour 10% du batch
        
        # Projection de sortie seulement pour les neurones actifs
        # h_active = h * is_active.float() # Version continue
        h_active = is_active.float() # Version purement binaire (Spiking like)
        
        out = torch.matmul(h_active, self.output_weights) # (batch, latent_dim)
        return out

    @torch.no_grad()
    def survival_update(self, reward: float):
        """
        Reward binaire externe: +1 (succès) ou -1 (échec).
        Met à jour l'état darwinien de chaque neurone indépendamment.
        """
        if self._last_active_mask is None:
            return
            
        active_mask = self._last_active_mask.bool()
        silent_mask = ~active_mask
        
        # 1. Règles de récompense/pénalité
        if reward > 0:
            self.survival_scores[active_mask] += 1
            self.dormancy_counters[active_mask] = 0
        else:
            self.survival_scores[active_mask] -= 0.5
            self.dormancy_counters[active_mask] += 2
            
        # 2. Le silence augmente la dormance
        self.dormancy_counters[silent_mask] += 1
        
        # 3. Mort et Renaissance
        dead_mask = self.dormancy_counters > self.death_threshold
        dead_indices = torch.where(dead_mask)[0]
        
        for i in dead_indices:
            self._rebirth(i)
            
        # Reset temp state
        self._last_active_mask = None

    def _rebirth(self, idx):
        """Réinitialise un neurone mort avec un nouveau vecteur aléatoire."""
        self.weights[idx] = torch.randn(self.latent_dim) * self.rebirth_noise
        self.bias[idx] = 0.0
        self.output_weights[idx] = torch.randn(self.latent_dim) * self.rebirth_noise
        self.survival_scores[idx] = 0
        self.dormancy_counters[idx] = 0

class SurvivalMoE(nn.Module):
    def __init__(self, latent_dim=512, num_experts=3):
        super().__init__()
        # Redefining LatentPredictor here to ensure it uses LayerNorm for Phase 5 batch_size=1
        class LocalLatentPredictor(nn.Module):
            def __init__(self, latent_dim=512, hidden_dim=2048, dropout=0.1):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim), 
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim)
                )
            def forward(self, z):
                return self.net(z)

        self.predictors = nn.ModuleList([
            LocalLatentPredictor(latent_dim=latent_dim, hidden_dim=2048, dropout=0.1) 
            for _ in range(num_experts)
        ])
        
        self.router = MultiPredictorEntropyRouter(self.predictors)
        
        self.experts = nn.ModuleList([
            SurvivalExpert(latent_dim=latent_dim) 
            for _ in range(num_experts)
        ])
        
        # Pour stocker l'expert utilisé afin de lui router le reward plus tard
        self._last_selected_experts = None

    def forward(self, z):
        # 1. Routing
        selected_experts, entropies = self.router(z)
        self._last_selected_experts = selected_experts
        
        # 2. Forward des experts
        out = torch.zeros_like(z)
        batch_size = z.size(0)
        
        for i in range(len(self.experts)):
            mask = (selected_experts == i)
            if mask.any():
                expert_z = z[mask]
                expert_out = self.experts[i](expert_z)
                out[mask] = expert_out
                
        return out
        
    def distribute_reward(self, reward: float):
        """
        Distribue le reward uniquement aux experts qui ont participé.
        (Dans une version plus granulaire, on pourrait passer un vecteur de rewards par sample).
        """
        if self._last_selected_experts is None:
            return
            
        active_expert_indices = torch.unique(self._last_selected_experts)
        
        for idx in active_expert_indices:
            self.experts[idx].survival_update(reward)

if __name__ == '__main__':
    print("Testing Survival MoE Architecture...")
    moe = SurvivalMoE(latent_dim=512, num_experts=3)
    
    # Mock data
    z = torch.randn(32, 512)
    
    # Forward pass (Routing + Darwinian Experts)
    out = moe(z)
    print(f"Output shape: {out.shape}")
    
    # Simulate a success reward
    print("Applying reward +1.0...")
    moe.distribute_reward(1.0)
    
    # Simulate a failure reward
    print("Applying reward -1.0...")
    moe.distribute_reward(-1.0)
    
    print("Done. Survival rules applied successfully without backprop.")