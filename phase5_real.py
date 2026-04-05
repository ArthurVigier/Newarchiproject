import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import subprocess
import tempfile
import json
import re
from datasets import load_dataset
import itertools

# Patch pour compatibilité transformers récente et GLM-4
_orig_getattr = nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return []
    return _orig_getattr(self, name)
nn.Module.__getattr__ = _patched_getattr

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
        self.encoder = StochasticTextEncoder(input_dim=llm_hidden_dim, latent_dim=latent_dim)
        self.moe = SurvivalMoE(latent_dim=latent_dim, num_experts=num_experts)
        self.projector = IntrospectionProjector(latent_dim=latent_dim, llm_embedding_dim=llm_hidden_dim)

    def forward(self, h_t):
        z_t, sigreg_loss = self.encoder(h_t)
        z_expert = self.moe(z_t)
        soft_token = self.projector(z_expert)
        return soft_token, sigreg_loss

def execute_lcb_reward(generated_code: str, input_output: str, timeout=5) -> float:
    """Reward binaire (Code) - exécution sandbox"""
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]

    try:
        io_data = json.loads(input_output)
        test_script = generated_code + "\n\nimport json\n"
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

def execute_math_reward(generated_text: str, target_str: str) -> float:
    """Reward binaire (Maths) - Exact match sur la réponse finale"""
    try:
        # GSM8K ground truth a le format "... #### 123"
        target_ans = target_str.split("####")[-1].strip()
        target_val = float(target_ans.replace(',', ''))
        
        # Extraction de tous les nombres générés par le modèle
        gen_nums = re.findall(r'-?\d+(?:\.\d+)?', generated_text.replace(',', ''))
        if not gen_nums:
            return -1.0
            
        # On suppose que la dernière valeur générée est la réponse finale
        gen_val = float(gen_nums[-1])
        
        if abs(gen_val - target_val) < 1e-5:
            return 1.0
        return -1.0
    except Exception:
        return -1.0

def load_mixed_datasets():
    print("Loading Mixed Curriculum Datasets...")
    ds_code = load_dataset("livecodebench/code_generation", split="test")
    ds_math = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds_code)} Code problems and {len(ds_math)} Math problems.")
    return ds_code, ds_math

def run_phase5_glm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_experts = 3
    
    model_name = "THUDM/glm-4-9b-chat"
    print(f"Loading Base LLM: {model_name} (Frozen)")
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Patch for GLM-4: remote code expects max_length but config has seq_length
    if not hasattr(config, 'max_length'):
        config.max_length = getattr(config, 'seq_length', 131072)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    llm_model.eval()
    for param in llm_model.parameters():
        param.requires_grad = False
        
    llm_hidden_dim = llm_model.config.hidden_size
    LAYER_IDX = 9 

    print("Initializing Scrupulous Experimental Architecture...")
    archi = ScrupulousArchitecture(llm_hidden_dim=llm_hidden_dim, latent_dim=512, num_experts=num_experts).to(device)
    
    backprop_params = list(archi.encoder.parameters()) + \
                      list(archi.projector.parameters()) + \
                      list(archi.moe.predictors.parameters())
    optimizer = optim.AdamW(backprop_params, lr=5e-5)
    
    ds_code, ds_math = load_mixed_datasets()
    
    print("\nStarting Double Learning Loop with Mixed Curriculum (Code + Math)...")
    
    # Tracking de spécialisation
    expert_usage = {"code": [0]*num_experts, "math": [0]*num_experts}
    
    # Générateur alterné
    def mixed_generator():
        for code_data, math_data in zip(ds_code, ds_math):
            yield ("code", code_data)
            yield ("math", math_data)

    for step, (task_type, data) in enumerate(mixed_generator()):
        if task_type == "code":
            prompt_text = data["question_content"]
            sys_msg = "You are an expert Python programmer. Solve the task with a function. Output code only."
        else:
            prompt_text = data["question"]
            sys_msg = "You are an expert mathematician. Solve the problem step by step and end with the final number."
            
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt_text}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            h_t = outputs.hidden_states[LAYER_IDX][:, -1, :].to(torch.float32)
            
        optimizer.zero_grad()
        soft_token, sigreg_loss = archi(h_t)
        
        # Tracking MoE
        active_expert = archi.moe._last_selected_experts[0].item()
        expert_usage[task_type][active_expert] += 1
        
        with torch.no_grad():
            word_embeddings = llm_model.get_input_embeddings()
            text_embeds = word_embeddings(inputs["input_ids"])
            soft_token_bf16 = soft_token.to(dtype=torch.bfloat16).unsqueeze(1)
            
            inputs_embeds = torch.cat([soft_token_bf16, text_embeds], dim=1)
            extra_mask = torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=device)
            extended_mask = torch.cat([extra_mask, inputs["attention_mask"]], dim=1)
            
            gen_outputs = llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                max_new_tokens=400 if task_type == "code" else 200,
                do_sample=True,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = gen_outputs[0][inputs_embeds.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        # Évaluation Binaire Externe
        if task_type == "code":
            reward = execute_lcb_reward(generated_text, data["input_output"])
        else:
            reward = execute_math_reward(generated_text, data["answer"])
        
        # BOUCLE A : Backprop
        policy_loss = -reward * torch.mean(soft_token**2) 
        total_loss = policy_loss + 0.1 * sigreg_loss
        total_loss.backward()
        optimizer.step()
        
        # BOUCLE B : Survie
        archi.moe.distribute_reward(reward)
        
        if (step + 1) % 10 == 0:
            print(f"\n--- Step {step+1:4d} ---")
            print(f"Task: {task_type.upper()} | Reward: {reward:2.0f} | Loss: {total_loss.item():.4f}")
            print(f"MoE Specialization Tracking:")
            print(f"  - Code Routed to Experts: {expert_usage['code']}")
            print(f"  - Math Routed to Experts: {expert_usage['math']}")

if __name__ == '__main__':
    run_phase5_glm()
