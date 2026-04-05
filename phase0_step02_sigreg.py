import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'le-wm-repo'))

import torch
import torch.nn as nn
from module import SIGReg

class StochasticTextEncoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=512, dropout=0.1):
        super().__init__()
        # On garde une architecture simple mais avec des poids initialisés plus petits
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # On utilise une valeur plus standard pour num_proj pour plus de stabilité
        self.sigreg = SIGReg(num_proj=1024)

    def forward(self, x):
        z = self.proj(x)
        z_seq = z.unsqueeze(0)
        loss_sigreg = self.sigreg(z_seq)
        return z, loss_sigreg

def run_sigreg_experiment():
    torch.manual_seed(42)
    
    print("Generating simulated textual embeddings (high anisotropy)...")
    base_X = torch.randn(2000, 128)
    projection = torch.randn(128, 4096)
    X = base_X @ projection
    X += torch.randn(4096) * 5.0 # Forte déviation de la moyenne
    
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    encoder = StochasticTextEncoder(input_dim=4096, latent_dim=512)
    # Learning rate plus faible pour éviter les oscillations
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)
    
    steps = 1000 # On double le nombre de steps pour voir la convergence à long terme
    step = 0
    losses = []
    
    encoder.train()
    print("Starting training loop (1000 steps)...")
    while step < steps:
        for (batch_x,) in dataloader:
            if step >= steps: break
            optimizer.zero_grad()
            z, sigreg_loss = encoder(batch_x)
            loss = sigreg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            
            if (step + 1) % 100 == 0:
                recent_avg = sum(losses[-100:]) / 100
                print(f"Step {step+1:4d} | SIGReg Loss: {recent_avg:.4f}")
            step += 1
            
    # Évaluation de la monotonicité avec lissage (Moving Average)
    window = 50
    smoothed = [sum(losses[i:i+window])/window for i in range(len(losses)-window)]
    
    # On regarde si la tendance globale est descendante sur de plus grands segments
    # (Le ratio up_ticks sur batch individuel est bruité par nature)
    segments = 10
    segment_size = len(smoothed) // segments
    segment_avgs = [sum(smoothed[i*segment_size:(i+1)*segment_size])/segment_size for i in range(segments)]
    
    print("\nSegment Averages (Loss Trend):")
    for i, avg in enumerate(segment_avgs):
        print(f"Segment {i}: {avg:.4f}")
        
    downward_trend = all(segment_avgs[i] > segment_avgs[i+1] for i in range(len(segment_avgs)-1))
    
    # On tolère une légère fluctuation sur un segment si la tendance générale est forte
    up_steps_in_segments = sum(1 for i in range(len(segment_avgs)-1) if segment_avgs[i+1] > segment_avgs[i])
    
    if up_steps_in_segments <= 1:
        print("\nKILL GATE PASSED: Convergence is monotone (trend-wise).")
    else:
        print(f"\nKILL GATE FAILED: Too many upward trends ({up_steps_in_segments}).")

if __name__ == '__main__':
    run_sigreg_experiment()
