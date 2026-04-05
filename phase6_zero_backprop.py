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
        # On désactive explicitement les gradients pour tous les paramètres
        self.encoder = StochasticTextEncoder(input_dim=llm_hidden_dim, latent_dim=latent_dim)
        for p in self.encoder.parameters(): p.requires_grad = False
            
        self.moe = SurvivalMoE(latent_dim=latent_dim, num_experts=num_experts)
        
        self.projector = IntrospectionProjector(latent_dim=latent_dim, llm_embedding_dim=llm_hidden_dim)
        for p in self.projector.parameters(): p.requires_grad = False
            
        # Paramètres d'évolution (Evolution Strategies)
        self.noise_std = 0.02     # Taille du pas de mutation
        self.learning_rate = 0.1  # Force de mise à jour après succès
        
        # Stockage de la dernière mutation appliquée
        self._last_mutation = []

    def forward(self, h_t):
        z_t, _ = self.encoder(h_t)
        z_expert = self.moe(z_t)
        soft_token = self.projector(z_expert)
        return soft_token

    @torch.no_grad()
    def mutate(self):
        """Applique une perturbation gaussienne aléatoire."""
        self._last_mutation = []
        for p in list(self.encoder.parameters()) + list(self.projector.parameters()):
            noise = torch.randn_like(p) * self.noise_std
            p.add_(noise)
            self._last_mutation.append((p, noise))

    @torch.no_grad()
    def evolution_step(self, reward: float):
        """Met à jour l'architecture basée sur le reward."""
        if reward > 0:
            # Succès : on renforce la mutation
            for p, noise in self._last_mutation:
                p.add_(noise * self.learning_rate)
        else:
            # Échec : on annule la mutation
            if self._last_mutation:
                for p, noise in self._last_mutation:
                    p.sub_(noise)
        self._last_mutation = []

    def save_checkpoint(self, path, step, expert_usage):
        checkpoint = {
            'step': step,
            'model_state_dict': self.state_dict(),
            'expert_usage': expert_usage
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint['step'], checkpoint['expert_usage']
        return 0, None

# --- SANDBOX EXECUTIONS ---
def execute_lcb_reward(generated_code: str, test_cases_str: str, timeout=5) -> float:
    if "```python" in generated_code: generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code: generated_code = generated_code.split("```")[1].split("```")[0]
    try:
        test_cases = json.loads(test_cases_str)
        if not test_cases: return -1.0
    except: return -1.0
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(generated_code)
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

def execute_math_reward(generated_text: str, target_str: str) -> float:
    try:
        target_ans = target_str.split("####")[-1].strip()
        target_val = float(target_ans.replace(',', ''))
        gen_nums = re.findall(r'-?\d+(?:\.\d+)?', generated_text.replace(',', ''))
        if not gen_nums: return -1.0
        gen_val = float(gen_nums[-1])
        if abs(gen_val - target_val) < 1e-5: return 1.0
        return -1.0
    except Exception: return -1.0

def execute_ai2_arc_reward(generated_text: str, correct_answer_key: str) -> float:
    """Reward pour allenai/ai2_arc (choix multiple)"""
    try:
        gen = generated_text.strip().upper()
        # On cherche la lettre de la réponse isolée
        match = re.search(r'\b' + correct_answer_key + r'\b', gen)
        if match: return 1.0
        return -1.0
    except: return -1.0

def load_triple_datasets():
    print("Loading Triple Curriculum Datasets (Code + Math + AI2_ARC)...")
    ds_code = load_dataset("livecodebench/code_generation", split="test")
    ds_math = load_dataset("openai/gsm8k", "main", split="test")
    # AI2 ARC Challenge
    ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"Loaded {len(ds_code)} Code, {len(ds_math)} Math, {len(ds_arc)} ARC problems.")
    return ds_code, ds_math, ds_arc

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

    print("Initializing 100% Zero-Backprop Evolutionary Architecture...")
    archi = ZeroBackpropArchitecture(llm_hidden_dim=llm_hidden_dim, latent_dim=512, num_experts=num_experts).to(device)
    
    ds_code, ds_math, ds_arc = load_triple_datasets()
    expert_usage = {"code": [0]*num_experts, "math": [0]*num_experts, "arc": [0]*num_experts}
    
    os.makedirs("/workspace/checkpoints", exist_ok=True)
    checkpoint_path = "/workspace/checkpoints/phase6_latest.pt"
    start_step, saved_usage = archi.load_checkpoint(checkpoint_path)
    if saved_usage: expert_usage = saved_usage
    
    def triple_generator():
        combined = zip(ds_code, ds_math, ds_arc)
        for _ in range(start_step // 3): next(combined, None)
        for code, math, arc in combined:
            yield ("code", code); yield ("math", math); yield ("arc", arc)

    print(f"\nStarting Darwinian Loop from Step {start_step}...")
    for relative_step, (task_type, data) in enumerate(triple_generator()):
        step = start_step + relative_step
        archi.mutate()
        
        if task_type == "code":
            prompt, sys_msg = data["question_content"], "You are an expert programmer. Solve the task by reading from stdin and writing to stdout. Output code only."
        elif task_type == "math":
            prompt, sys_msg = data["question"], "You are an expert mathematician. Solve the problem step by step and end with #### number."
        else: # AI2_ARC
            choices_text = ""
            for label, text in zip(data['choices']['label'], data['choices']['text']):
                choices_text += f"{label}: {text}\n"
            prompt = f"Question: {data['question']}\nChoices:\n{choices_text}\nAnswer:"
            sys_msg = "You are a science expert. Choose the correct option label (A, B, C, or D). Output only the label."
            
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            h_t = outputs.hidden_states[LAYER_IDX][:, -1, :].to(torch.float32)
            soft_token = archi(h_t)
            
            active_expert = archi.moe._last_selected_experts[0].item()
            expert_usage[task_type][active_expert] += 1
            
            word_embeddings = llm_model.get_input_embeddings()
            text_embeds = word_embeddings(inputs["input_ids"])
            soft_token_bf16 = soft_token.to(dtype=torch.bfloat16).view(1, 1, -1)
            
            inputs_embeds = torch.cat([soft_token_bf16, text_embeds], dim=1)
            extended_mask = torch.cat([torch.ones((1, 1), device=device), inputs["attention_mask"]], dim=1)
            
            gen_outputs = llm_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=extended_mask, 
                max_new_tokens=400 if task_type == "code" else 100, 
                do_sample=True, 
                temperature=0.2, 
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(gen_outputs[0][inputs_embeds.shape[1]:], skip_special_tokens=True)
            
        if task_type == "code": reward = execute_lcb_reward(generated_text, data["public_test_cases"])
        elif task_type == "math": reward = execute_math_reward(generated_text, data["answer"])
        else: reward = execute_ai2_arc_reward(generated_text, data["answerKey"])

        archi.evolution_step(reward)
        archi.moe.distribute_reward(reward)
        
        if (step + 1) % 50 == 0: archi.save_checkpoint(checkpoint_path, step + 1, expert_usage)
        if (step + 1) % 15 == 0:
            print(f"\n--- Step {step+1} ---")
            print(f"Task: {task_type.upper()} | Reward: {reward} | [ZERO BACKPROP]")
            print(f"Usage: Code {expert_usage['code']} | Math {expert_usage['math']} | ARC {expert_usage['arc']}")

if __name__ == '__main__':
    run_phase6_zero_backprop()
