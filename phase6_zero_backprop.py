import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import subprocess
import tempfile
import json
import re
from datasets import load_dataset
import itertools

# Configuration du chemin pour les modules locaux
sys.path.append(os.getcwd())
try:
    from phase1_encoder import StochasticTextEncoder
    from phase2_survival_moe import SurvivalMoE
    from phase4_introspection import IntrospectionProjector
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

class ZeroBackpropArchitecture(nn.Module):
    """Architecture 100% Darwinienne/Evolutionnaire. AUCUN GRADIENT."""
    def __init__(self, llm_hidden_dim=5120, latent_dim=512, num_experts=3):
        super().__init__()
        self.encoder = StochasticTextEncoder(input_dim=llm_hidden_dim, latent_dim=latent_dim)
        for p in self.encoder.parameters(): p.requires_grad = False
            
        self.moe = SurvivalMoE(latent_dim=latent_dim, num_experts=num_experts)
        
        self.projector = IntrospectionProjector(latent_dim=latent_dim, llm_embedding_dim=llm_hidden_dim)
        for p in self.projector.parameters(): p.requires_grad = False
            
        self.noise_std = 0.02
        self.learning_rate = 0.1
        self._last_mutation = []

    def forward(self, h_t):
        z_t, _ = self.encoder(h_t)
        z_expert = self.moe(z_t)
        soft_token = self.projector(z_expert)
        return soft_token

    @torch.no_grad()
    def mutate(self):
        self._last_mutation = []
        for p in list(self.encoder.parameters()) + list(self.projector.parameters()):
            noise = torch.randn_like(p) * self.noise_std
            p.add_(noise)
            self._last_mutation.append((p, noise))

    @torch.no_grad()
    def evolution_step(self, reward: float):
        if reward > 0:
            for p, noise in self._last_mutation:
                p.add_(noise * self.learning_rate)
        else:
            if self._last_mutation:
                for p, noise in self._last_mutation:
                    p.sub_(noise)
        self._last_mutation = []

    def save_checkpoint(self, path, step, expert_usage, task_stats):
        checkpoint = {
            'step': step,
            'model_state_dict': self.state_dict(),
            'expert_usage': expert_usage,
            'task_stats': task_stats
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint['step'], checkpoint.get('expert_usage'), checkpoint.get('task_stats')
        return 0, None, None

# --- PURE CODE SANDBOX EXECUTIONS ---
def extract_code(text: str) -> str:
    if "```python" in text: return text.split("```python")[1].split("```")[0]
    elif "```" in text: return text.split("```")[1].split("```")[0]
    return text

def execute_lcb_reward(generated_code: str, test_cases_str: str, timeout=5) -> float:
    """Sandbox for LiveCodeBench (stdin/stdout)"""
    code = extract_code(generated_code)
    try:
        test_cases = json.loads(test_cases_str)
        if not test_cases: return -1.0
    except: return -1.0
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_filename = f.name
    reward = 1.0
    try:
        for tc in test_cases[:2]:
            result = subprocess.run([sys.executable, temp_filename], input=tc.get("input", ""), capture_output=True, timeout=timeout, text=True)
            if result.returncode != 0 or result.stdout.strip() != tc.get("output", "").strip():
                reward = -1.0; break
    except: reward = -1.0
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)
    return reward

def execute_mbpp_reward(generated_code: str, assertions: list, timeout=5) -> float:
    """Sandbox for MBPP (Python Assertions)"""
    code = extract_code(generated_code)
    full_script = code + "\n\n" + "\n".join(assertions) + "\nprint('OK')"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_script)
        temp_filename = f.name
        
    try:
        result = subprocess.run([sys.executable, temp_filename], capture_output=True, timeout=timeout, text=True)
        if result.returncode == 0 and "OK" in result.stdout:
            return 1.0
        return -1.0
    except: return -1.0
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

def execute_bigcodebench_reward(generated_code: str, test_setup: str, test_call: str, timeout=10) -> float:
    """Sandbox for BigCodeBench (Complex functions with libraries)"""
    code = extract_code(generated_code)
    # Combine code, setup, and the test execution
    full_script = f"{code}\n\n{test_setup}\n\ntry:\n    {test_call}\n    print('OK')\nexcept Exception as e:\n    print('FAIL')"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_script)
        temp_filename = f.name
        
    try:
        # Note: In a real environment, you'd need the 139+ libraries installed.
        # This will likely fail (-1) if the pod lacks the specific library required by the task.
        result = subprocess.run([sys.executable, temp_filename], capture_output=True, timeout=timeout, text=True)
        if result.returncode == 0 and "OK" in result.stdout:
            return 1.0
        return -1.0
    except: return -1.0
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

def execute_apps_reward(generated_code: str, input_output_json: str, timeout=5) -> float:
    """Sandbox for APPS (Competitive Programming, strict stdin/stdout)"""
    code = extract_code(generated_code)
    try:
        io_data = json.loads(input_output_json)
        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])
        if not inputs or not outputs or len(inputs) != len(outputs):
            return -1.0
    except: return -1.0

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_filename = f.name
        
    reward = 1.0
    try:
        # Check first 2 hidden test cases to save time in the loop
        for inp, exp_out in zip(inputs[:2], outputs[:2]):
            result = subprocess.run([sys.executable, temp_filename], input=inp, capture_output=True, timeout=timeout, text=True)
            if result.returncode != 0 or result.stdout.strip() != exp_out.strip():
                reward = -1.0
                break
    except: reward = -1.0
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)
    return reward

def load_extended_code_datasets():
    print("Loading EXTENDED PURE CODE Curriculum Datasets...")
    
    ds_lcb = load_dataset("livecodebench/code_generation", split="test")
    print(f"Loaded {len(ds_lcb)} LiveCodeBench problems (Algorithmic/IO)")
    
    ds_mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    print(f"Loaded {len(ds_mbpp)} MBPP problems (Basic Python/Assertions)")
    
    # BigCodeBench: Focuses on real-world library usage (Pandas, API calls, etc.)
    ds_bcb = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    print(f"Loaded {len(ds_bcb)} BigCodeBench problems (Practical Dev/Libraries)")
    
    # APPS: Hardcore competitive programming
    ds_apps = load_dataset("codeparrot/apps", split="test")
    print(f"Loaded {len(ds_apps)} APPS problems (Competitive Programming)")
    
    return ds_lcb, ds_mbpp, ds_bcb, ds_apps

def run_phase6_zero_backprop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_experts = 3
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    llm_model.eval()
    for param in llm_model.parameters(): param.requires_grad = False
        
    llm_hidden_dim = llm_model.config.hidden_size # 5120
    LAYER_IDX = 9

    print("Initializing 100% Zero-Backprop Evolutionary Architecture (PURE CODE FOCUS)...")
    archi = ZeroBackpropArchitecture(llm_hidden_dim=llm_hidden_dim, latent_dim=512, num_experts=num_experts).to(device)
    
    ds_lcb, ds_mbpp, ds_bcb, ds_apps = load_extended_code_datasets()
    
    # Tracking
    expert_usage = {
        "lcb": [0]*num_experts, 
        "mbpp": [0]*num_experts,
        "bcb": [0]*num_experts,
        "apps": [0]*num_experts
    }
    task_stats = {
        "lcb": {"attempts": 0, "success": 0},
        "mbpp": {"attempts": 0, "success": 0},
        "bcb": {"attempts": 0, "success": 0},
        "apps": {"attempts": 0, "success": 0}
    }
    
    os.makedirs("/workspace/checkpoints", exist_ok=True)
    checkpoint_path = "/workspace/checkpoints/phase6_extended_code.pt"
    start_step, saved_usage, saved_stats = archi.load_checkpoint(checkpoint_path)
    if saved_usage: expert_usage = saved_usage
    if saved_stats: task_stats = saved_stats
    
    def code_generator():
        # Cycle entrelacé des 4 datasets
        combined = zip(ds_lcb, ds_mbpp, ds_bcb, ds_apps)
        for _ in range(start_step // 4): next(combined, None)
        for lcb, mbpp, bcb, apps in combined:
            yield ("lcb", lcb)
            yield ("mbpp", mbpp)
            yield ("bcb", bcb)
            yield ("apps", apps)

    print(f"\nStarting Extended Darwinian Loop from Step {start_step}...")
    for relative_step, (task_type, data) in enumerate(code_generator()):
        step = start_step + relative_step
        archi.mutate()
        
        # --- PROMPT PREPARATION ---
        if task_type == "lcb":
            prompt = data["question_content"]
            sys_msg = "You are an expert programmer. Solve the algorithmic task by reading from stdin and writing to stdout. Output Python code only."
        elif task_type == "mbpp":
            prompt = f"Task: {data['text']}\nEnsure the function is named exactly as expected by these assertions: {data['test_list']}"
            sys_msg = "You are an expert Python programmer. Write a simple Python function to solve the task. Output Python code only."
        elif task_type == "bcb":
            prompt = f"Task: {data['complete_prompt']}\nWrite the implementation for this function using the appropriate libraries."
            sys_msg = "You are an expert software engineer. Complete the Python function using standard libraries. Output Python code only."
        else: # apps
            prompt = data["question"]
            sys_msg = "You are a competitive programming expert. Write a Python script that reads from stdin and prints to stdout. Output Python code only."
            
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        # --- EXECUTION ---
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            h_t = outputs.hidden_states[LAYER_IDX][:, -1, :].to(torch.float32)
            soft_token = archi(h_t)
            
            active_expert = archi.moe._last_selected_experts[0].item()
            expert_usage[task_type][active_expert] += 1
            task_stats[task_type]["attempts"] += 1
            
            word_embeddings = llm_model.get_input_embeddings()
            text_embeds = word_embeddings(inputs["input_ids"])
            soft_token_bf16 = soft_token.to(dtype=torch.bfloat16).view(1, 1, -1)
            
            inputs_embeds = torch.cat([soft_token_bf16, text_embeds], dim=1)
            extended_mask = torch.cat([torch.ones((1, 1), device=device), inputs["attention_mask"]], dim=1)
            
            # Ajustement des tokens selon la complexité attendue
            max_tokens = 500 if task_type in ["lcb", "apps", "bcb"] else 200
            
            gen_outputs = llm_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=extended_mask, 
                max_new_tokens=max_tokens, 
                do_sample=True, 
                temperature=0.2, 
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(gen_outputs[0][inputs_embeds.shape[1]:], skip_special_tokens=True)
            
        # --- REWARD EVALUATION ---
        if task_type == "lcb": 
            reward = execute_lcb_reward(generated_text, data["public_test_cases"])
        elif task_type == "mbpp": 
            reward = execute_mbpp_reward(generated_text, data["test_list"])
        elif task_type == "bcb":
            reward = execute_bigcodebench_reward(generated_text, data["test_setup"], data["test"])
        else: # apps
            reward = execute_apps_reward(generated_text, data["input_output"])

        if reward > 0:
            task_stats[task_type]["success"] += 1

        # --- DARWINIAN UPDATE ---
        archi.evolution_step(reward)
        archi.moe.distribute_reward(reward)
        
        # --- LOGGING ---
        if (step + 1) % 50 == 0: 
            archi.save_checkpoint(checkpoint_path, step + 1, expert_usage, task_stats)
            
        if (step + 1) % 10 == 0:
            print(f"\n[{step+1:4d}] Task: {task_type.upper():4s} | Reward: {reward:2.0f} | [ZERO BACKPROP]")
            print(f"Spécialisation MoE (Route par Expert):")
            print(f"  - MBPP (Python Base)   : {expert_usage['mbpp']} (Acc: {task_stats['mbpp']['success']}/{task_stats['mbpp']['attempts']})")
            print(f"  - BigCodeBench (Libs)  : {expert_usage['bcb']} (Acc: {task_stats['bcb']['success']}/{task_stats['bcb']['attempts']})")
            print(f"  - LiveCodeBench (Algo) : {expert_usage['lcb']} (Acc: {task_stats['lcb']['success']}/{task_stats['lcb']['attempts']})")
            print(f"  - APPS (Compet. Prog.) : {expert_usage['apps']} (Acc: {task_stats['apps']['success']}/{task_stats['apps']['attempts']})")

if __name__ == '__main__':
    run_phase6_zero_backprop()
