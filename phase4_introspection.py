import torch
import torch.nn as nn

class IntrospectionProjector(nn.Module):
    """
    Projecteur Introspectif (Phase 4).
    Prend le vecteur latent stochastique généré par les experts de survie (z_expert)
    et le projette dans l'espace d'embedding du LLM pour qu'il soit injecté
    comme un "token virtuel" (soft prompt).
    """
    def __init__(self, latent_dim=512, llm_embedding_dim=4096, hidden_dim=2048):
        super().__init__()
        
        # Ce composant EST entraîné par backprop (contrairement aux experts).
        # Il sert de pont traducteur entre "l'intuition darwinienne" et le langage du LLM.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # Projection vers la dimension d'embedding exacte du modèle cible
            nn.Linear(hidden_dim, llm_embedding_dim)
        )

    def forward(self, z_expert):
        """
        z_expert: Tensor de forme (batch_size, latent_dim)
        Retourne: Tensor de forme (batch_size, 1, llm_embedding_dim)
                  qui peut être concaténé avec les embeddings des tokens d'entrée.
        """
        # (batch, llm_embedding_dim)
        projected = self.net(z_expert)
        
        # Ajout de la dimension de séquence (batch, seq_len=1, llm_embedding_dim)
        # pour correspondre à l'attente de inputs_embeds dans HuggingFace
        soft_token = projected.unsqueeze(1)
        
        return soft_token

# Exemple d'intégration (conceptuel) avec HuggingFace pour l'injection
def inject_soft_token_into_llm(llm_model, tokenizer, prompt_text, soft_token, device='cpu'):
    """
    Fonction utilitaire montrant comment injecter le token généré par le projecteur
    dans le LLM de base figé (frozen).
    """
    llm_model.eval() # Le LLM de base reste figé
    
    # 1. Tokenisation du texte standard
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 2. Récupération de la couche d'embedding du LLM
    # (Dépend du modèle, pour Qwen/Llama c'est souvent model.model.embed_tokens)
    word_embeddings = llm_model.get_input_embeddings()
    
    # 3. Conversion des IDs de texte en vecteurs d'embedding (batch, seq_len, embed_dim)
    with torch.no_grad():
        text_embeds = word_embeddings(input_ids)
    
    # 4. INJECTION : Concaténation du token d'introspection en POSITION 0
    # On ajoute le "z_expert" au tout début du contexte, comme une instruction globale
    inputs_embeds = torch.cat([soft_token, text_embeds], dim=1)
    
    # 5. Ajustement de l'attention mask (on ajoute 1 token "toujours vu" au début)
    # Forme initiale du mask: (batch, seq_len)
    batch_size = attention_mask.shape[0]
    extra_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
    extended_attention_mask = torch.cat([extra_mask, attention_mask], dim=1)
    
    # 6. Génération (ou forward pass) utilisant les embeddings modifiés
    # Note: On ne passe plus `input_ids`, mais `inputs_embeds`
    outputs = llm_model(
        inputs_embeds=inputs_embeds,
        attention_mask=extended_attention_mask,
        output_hidden_states=True
    )
    
    return outputs

if __name__ == '__main__':
    print("Testing Introspection Projector Architecture...")
    
    # Dimensions pour Qwen3-4B
    LATENT_DIM = 512
    LLM_EMBED_DIM = 4096
    BATCH_SIZE = 4
    
    projector = IntrospectionProjector(latent_dim=LATENT_DIM, llm_embedding_dim=LLM_EMBED_DIM)
    
    # Simule l'output du SurvivalMoE
    z_expert_mock = torch.randn(BATCH_SIZE, LATENT_DIM)
    
    # Projection
    soft_token = projector(z_expert_mock)
    
    print(f"Input shape (from MoE): {z_expert_mock.shape}")
    print(f"Output shape (to LLM):  {soft_token.shape}")
    print("\nArchitectural test passed. Projector output is ready for inputs_embeds concatenation.")
