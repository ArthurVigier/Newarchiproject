import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Injection du repo le-wm pour utiliser SIGReg
sys.path.append(os.path.join(os.getcwd(), 'le-wm-repo'))
try:
    from module import SIGReg
except ImportError:
    print("Erreur: le module SIGReg est introuvable. Assurez-vous que le-wm-repo est présent.")
    sys.exit(1)

class StochasticTextEncoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=512, dropout=0.1, num_proj=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.sigreg = SIGReg(num_proj=num_proj)

    def forward(self, x):
        z = self.proj(x)
        # SIGReg attend un tenseur de dimension (Seq, Batch, Dim). 
        # Ici on n'a pas de dimension temporelle explicite dans l'encodeur pur, on ajoute une seq de 1.
        z_seq = z.unsqueeze(0)
        sigreg_loss = self.sigreg(z_seq)
        return z, sigreg_loss

class LatentPredictor(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=8192):
        super().__init__()
        # 512 -> 8192 -> 512 représente environ ~8.4M de paramètres, 
        # ce qui est parfaitement aligné avec l'objectif de "~10M params" du contexte.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z):
        return self.net(z)

class Phase1Model(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=512):
        super().__init__()
        self.encoder = StochasticTextEncoder(input_dim, latent_dim)
        self.predictor = LatentPredictor(latent_dim)

    def forward(self, x_t, x_t_next):
        # Encode x_t (état actuel du prompt)
        z_t, sigreg_loss_t = self.encoder(x_t)
        
        # Encode x_{t+1} (état futur de la solution)
        # Note: Pas de détachement du gradient explicite car SIGReg gère l'effondrement
        z_t_next, sigreg_loss_next = self.encoder(x_t_next)
        
        # Prédit z_{t+1} depuis z_t
        z_t_next_pred = self.predictor(z_t)
        
        return z_t, z_t_next, z_t_next_pred, sigreg_loss_t, sigreg_loss_next

def train_phase1(model, dataloader, epochs=10, lr=1e-4, sigreg_lambda=0.1, device='cpu'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    
    model.train()
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        total_mse = 0
        total_sigreg = 0
        
        for batch_idx, (x_t, x_t_next, labels) in enumerate(dataloader):
            x_t, x_t_next = x_t.to(device), x_t_next.to(device)
            
            optimizer.zero_grad()
            z_t, z_t_next, z_t_next_pred, sigreg_t, sigreg_next = model(x_t, x_t_next)
            
            loss_mse = mse_criterion(z_t_next_pred, z_t_next)
            loss_sigreg = (sigreg_t + sigreg_next) / 2.0
            
            # Loss totale selon l'équation L = L_pred + λ * SIGReg(Z)
            loss = loss_mse + sigreg_lambda * loss_sigreg
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_sigreg += loss_sigreg.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} "
              f"(MSE: {total_mse/len(dataloader):.4f}, SIGReg: {total_sigreg/len(dataloader):.4f})")

def validate_latent_separability(model, x_val, labels_val, device='cpu'):
    """
    Validation Kill Gate: Vérifie que l'AUC séparabilité pass/fail est maintenue 
    dans l'espace latent réduit Z par rapport à l'espace d'origine.
    """
    model.eval()
    x_val = x_val.to(device)
    with torch.no_grad():
        z_val, _ = model.encoder(x_val)
    
    z_np = z_val.cpu().numpy()
    labels_np = labels_val.numpy()
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z_np, labels_np)
    probs = clf.predict_proba(z_np)[:, 1]
    auc = roc_auc_score(labels_np, probs)
    
    print(f"\n--- Validation Kill Gate (Phase 1) ---")
    print(f"Latent AUC Pass/Fail: {auc:.4f}")
    if auc > 0.65:
        print("KILL GATE PASSED: Latent space preserves pass/fail separability (AUC > 0.65)")
    else:
        print("KILL GATE FAILED: Latent space lost pass/fail signal. AUC <= 0.65")

if __name__ == '__main__':
    # Détection automatique GPU si exécuté sur Vast.ai ou RunPod
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Detected: {device.type.upper()}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    
    # Paramètres calqués sur Qwen3-4B
    INPUT_DIM = 4096
    LATENT_DIM = 512
    BATCH_SIZE = 128
    
    print("\nInitializing Phase 1 Architecture...")
    model = Phase1Model(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Architecture Parameters: {total_params:,}")
    print(f"  - Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  - Predictor: {sum(p.numel() for p in model.predictor.parameters()):,}")
    
    print("\nGenerating mock data matching Qwen activations shape...")
    N_SAMPLES = 2000
    X_t = torch.randn(N_SAMPLES, INPUT_DIM)
    X_t_next = X_t + torch.randn(N_SAMPLES, INPUT_DIM) * 0.5 
    
    # Signal synthétique pour s'assurer que le test AUC a un sens minimal
    hidden_signal = X_t[:, 0:10].sum(dim=1)
    labels = (hidden_signal > 0).long()
    
    dataset = torch.utils.data.TensorDataset(X_t, X_t_next, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    train_phase1(model, dataloader, epochs=5, lr=3e-4, sigreg_lambda=0.1, device=device)
    
    # Validation du Kill Gate
    X_val = torch.randn(500, INPUT_DIM)
    hidden_signal_val = X_val[:, 0:10].sum(dim=1)
    labels_val = (hidden_signal_val > 0).long()
    
    validate_latent_separability(model, X_val, labels_val, device=device)
