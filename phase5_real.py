import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import subprocess
import tempfile
import json
import urllib.request

sys.path.append(os.getcwd())
try:
    from phase1_encoder import StochasticTextEncoder
    from phase2_survival_moe import SurvivalMoE
    from phase4_introspection import IntrospectionProjector
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

class FullArchitecture(nn.Module):
    def __init__(self, llm_hidden_dim=3584, latent_dim=512, num_experts=3):
        super().__init__()
        self.encoder = StochasticTextEncoder(input_dim=llm_hidden_dim, latent_dim=latent_dim)
        self.moe = SurvivalMoE(latent_dim=latent_dim, num_experts=num_experts)
        self.projector = IntrospectionProjector(latent_dim=latent_dim, llm_embedding_dim=llm_hidden_dim)

def execute_and_reward(generated_code: str, test_assertions: list, timeout=5) -> float:
    """
    Exécute le code généré dans un bac à sable (subprocess) avec les tests unitaires.
    Retourne +1.0 si tous les tests passent, -1.0 en cas d'échec, erreur ou timeout.
    """
    # Nettoyage basique du markdown généré par le LLM
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(generated_code + "\n\n")
        f.write("\n".join(test_assertions))
        temp_filename = f.name
        
    try:
        result = subprocess.run(
            [sys.executable, temp_filename],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        if result.returncode == 0:
            reward = 1.0  # Succès total
        else:
            reward = -1.0 # Erreur d'exécution ou test échoué
    except subprocess.TimeoutExpired:
        reward = -1.0 # Boucle infinie
    except Exception:
        reward = -1.0
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    return reward

def load_mbpp_dataset(limit=500):
    """
    Charge le dataset MBPP (Mostly Basic Python Problems) de Google Research.
    Très similaire à BigCodeBench en termes de structure (Prompt + Tests unitaires).
    """
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    print("Downloading dataset (MBPP)...")
    dataset = []
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            for line in response:
                obj = json.loads(line)
                dataset.append(obj)
                if len(dataset) >= limit:
                    break
        print(f"Loaded {len(dataset)} problems.")
    except Exception as e:
        print(f"Could not load MBPP: {e}")
    return dataset

def run_real_phase5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing REAL Phase 5 on {device.type.upper()}...")
    
    # Qwen2.5-Coder-7B
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    print(f"Loading Base LLM: {model_name} (Frozen)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    llm_model.eval()
    for param in llm_model.parameters():
        param.requires_grad = False
        
    llm_hidden_dim = llm_model.config.hidden_size

    print("Loading Experimental Architecture...")
    archi = FullArchitecture(llm_hidden_dim=llm_hidden_dim, latent_dim=512, num_experts=3)
    archi = archi.to(device)
    
    optimizer = optim.AdamW([p for p in archi.parameters() if p.requires_grad], lr=5e-5)
    LAYER_IDX = 9 
    
    dataset = load_mbpp_dataset(limit=1000) # Charger 1000 problèmes
    
    print("\nStarting REAL Training Loop...")
    metrics_history = []
    
    for step, data in enumerate(dataset):
        prompt_text = data["text"]
        test_list = data["test_list"]
        
        # Format the prompt as expected by instruction-tuned Qwen
        sys_prompt = "You are an expert Python programmer. Write the Python function to solve the following task. Only output valid Python code, no explanations."
        full_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        
        optimizer.zero_grad()
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            h_t = outputs.hidden_states[LAYER_IDX][:, -1, :].to(torch.float32)
            
        # The Custom Architecture Pipeline
        z_t, sigreg_loss = archi.encoder(h_t)
        z_expert = archi.moe(z_t)
        soft_token = archi.projector(z_expert)
        
        # Injection and Generation
        with torch.no_grad():
            word_embeddings = llm_model.get_input_embeddings()
            text_embeds = word_embeddings(inputs["input_ids"])
            soft_token_bf16 = soft_token.to(dtype=torch.bfloat16)
            inputs_embeds = torch.cat([soft_token_bf16, text_embeds], dim=1)
            
            extra_mask = torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=device)
            extended_mask = torch.cat([extra_mask, inputs["attention_mask"]], dim=1)
            
            gen_outputs = llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                max_new_tokens=200, # Vraie génération de code
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.2, # Low temp pour le code
                do_sample=True
            )
            # Ne récupérer que la partie générée (ignorer le prompt)
            generated_ids = gen_outputs[0][inputs_embeds.shape[1] - 1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        # Evaluate using real subprocess Sandbox
        reward = execute_and_reward(generated_text, test_list)
        metrics_history.append(1 if reward > 0 else 0)
        
        # Double Update 
        # Loss policy classique: -reward * (norme du token / ou sortie du projecteur)
        # Ceci force le projecteur à "pousser" plus fort si le code est bon
        policy_loss = -reward * soft_token.norm() 
        total_loss = policy_loss + 0.1 * sigreg_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(archi.parameters(), 1.0)
        optimizer.step()
        
        # Darwinian update on MoE
        archi.moe.distribute_reward(reward)
        
        if (step + 1) % 10 == 0:
            recent_acc = sum(metrics_history[-10:]) / 10.0
            print(f"Step {step+1:4d} | Recent Acc: {recent_acc*100:.1f}% | Loss: {total_loss.item():.4f}")

    print("\nReal Phase 5 Training Completed.")

if __name__ == '__main__':
    run_real_phase5()
