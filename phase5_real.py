import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import subprocess
import tempfile
import json
from datasets import load_dataset

# Configuration du chemin pour les modules locaux
sys.path.append(os.getcwd())
try:
    from phase1_encoder import StochasticTextEncoder
    from phase2_survival_moe import SurvivalMoE
    from phase4_introspection import IntrospectionProjector
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

class ScrupulousArchitecture(nn.Module):
    def __init__(self, llm_hidden_dim=4096, latent_dim=512, num_experts=3):
        super().__init__()
        # 1. Encodeur Stochastique (SIGReg intégré)
        self.encoder = StochasticTextEncoder(input_dim=llm_hidden_dim, latent_dim=latent_dim)
        
        # 2. Router Entropique + Experts à Survie Darwinienne
        # Le MoE contient déjà le EntropyRouter avec MC Dropout (5 passes)
        self.moe = SurvivalMoE(latent_dim=latent_dim, num_experts=num_experts)
        
        # 3. Projecteur Introspectif (Entraîné par backprop)
        self.projector = IntrospectionProjector(latent_dim=latent_dim, llm_embedding_dim=llm_hidden_dim)

    def forward(self, h_t):
        # h_t: activations de la couche 9 (batch, dim)
        
        # A. Encodage avec SIGReg
        z_t, sigreg_loss = self.encoder(h_t)
        
        # B. Routing Entropique et Traitement par Expert Survivaliste
        # L'expert sélectionné est celui avec la plus faible variance (MC Dropout)
        z_expert = self.moe(z_t)
        
        # C. Projection vers l'espace d'embedding du LLM
        soft_token = self.projector(z_expert)
        
        return soft_token, sigreg_loss

def execute_lcb_reward(generated_code: str, input_output: str, timeout=5) -> float:
    """Reward binaire, déterministe, externe - pas de reward hacking."""
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]

    try:
        io_data = json.loads(input_output)
        test_script = generated_code + "\n\n"
        test_script += "import json\n"
        test_script += f"inputs = {io_data.get('inputs', [])}\n"
        test_script += f"expected_outputs = {io_data.get('outputs', [])}\n"
        test_script += f"fn_name = '{io_data.get('fn_name', '')}'\n"
        test_script += """
if fn_name and fn_name in globals():
    func = globals()[fn_name]
    for i in range(len(inputs)):
        inp = inputs[i]
        expected = expected_outputs[i]
        res = func(*inp) if isinstance(inp, list) else func(inp)
        if res != expected:
            import sys
            sys.exit(1)
    print("All tests passed!")
else:
    import sys
    sys.exit(1)
"""
    except Exception:
        return -1.0

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_filename = f.name
        
    try:
        result = subprocess.run([sys.executable, temp_filename], capture_output=True, timeout=timeout, text=True)
        reward = 1.0 if (result.returncode == 0 and "All tests passed!" in result.stdout) else -1.0
    except Exception:
        reward = -1.0
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    return reward

def run_phase5_glm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODÈLE GLM RÉCENT (GLM-4-9B-Chat)
    # Performant en coding, tient largement sur A100 80GB
    model_name = "THUDM/glm-4-9b-chat"
    print(f"Loading Base LLM: {model_name} (Frozen)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    llm_model.eval()
    for param in llm_model.parameters():
        param.requires_grad = False
        
    llm_hidden_dim = llm_model.config.hidden_size # 4096 pour GLM-4-9B
    LAYER_IDX = 9 # Couche d'intérêt spécifiée par Arthur

    print("Initializing Scrupulous Experimental Architecture...")
    archi = ScrupulousArchitecture(llm_hidden_dim=llm_hidden_dim, latent_dim=512, num_experts=3).to(device)
    
    # Séparation des paramètres pour la double boucle
    # Backprop: Encoder, Projector, Predictors du Router
    backprop_params = list(archi.encoder.parameters()) + \
                      list(archi.projector.parameters()) + \
                      list(archi.moe.predictors.parameters())
    
    optimizer = optim.AdamW(backprop_params, lr=5e-5)
    
    dataset = load_dataset("livecodebench/code_generation", split="test")
    
    print("\nStarting Double Learning Loop (Backprop + Darwinian Survival)...")
    
    for step, data in enumerate(dataset):
        prompt_text = data["question_content"]
        input_output_str = data["input_output"]
        
        # Formatage GLM-4
        messages = [{"role": "system", "content": "You are an expert Python programmer. Solve the task with a function. Output code only."},
                    {"role": "user", "content": prompt_text}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        # 1. Extraction des activations (FROZEN LLM)
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            h_t = outputs.hidden_states[LAYER_IDX][:, -1, :].to(torch.float32)
            
        # 2. Pipeline Architecture Expérimentale
        optimizer.zero_grad()
        soft_token, sigreg_loss = archi(h_t)
        
        # 3. Injection Introspective (Token en position 0)
        with torch.no_grad():
            word_embeddings = llm_model.get_input_embeddings()
            text_embeds = word_embeddings(inputs["input_ids"]) # (1, seq, dim)
            soft_token_bf16 = soft_token.to(dtype=torch.bfloat16).unsqueeze(1) # (1, 1, dim)
            
            inputs_embeds = torch.cat([soft_token_bf16, text_embeds], dim=1)
            extra_mask = torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=device)
            extended_mask = torch.cat([extra_mask, inputs["attention_mask"]], dim=1)
            
            # Génération avec injection
            gen_outputs = llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            # On retire les tokens d'entrée (incluant le soft token) pour décoder la réponse
            generated_ids = gen_outputs[0][inputs_embeds.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        # 4. Évaluation Binaire Externe
        reward = execute_lcb_reward(generated_text, input_output_str)
        
        # 5. BOUCLE A : MISE À JOUR PAR BACKPROPAGATION
        # Utilisation d'un signal de type policy gradient simplifié pour le projecteur
        # L = -reward * log_prob (ici simulé par la norme pour orienter la projection)
        # + régularisation SIGReg pour l'encodeur
        policy_loss = -reward * torch.mean(soft_token**2) 
        total_loss = policy_loss + 0.1 * sigreg_loss
        total_loss.backward()
        optimizer.step()
        
        # 6. BOUCLE B : MISE À JOUR DARWINIENNE (ZÉRO GRADIENT)
        # Mise à jour locale des scores de survie, dormance et renaissances
        archi.moe.distribute_reward(reward)
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1:4d} | Reward: {reward:2.0f} | Loss: {total_loss.item():.4f} | GLM-4 Active")

if __name__ == '__main__':
    run_phase5_glm()
