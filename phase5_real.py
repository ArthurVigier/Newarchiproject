import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import subprocess
import tempfile
import json

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

def execute_lcb_reward(generated_code: str, input_output: str, timeout=5) -> float:
    """
    Exécute le code généré contre les tests unitaires fournis par LiveCodeBench.
    LiveCodeBench fournit souvent les I/O sous forme de string JSON qu'il faut parser.
    """
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]

    # Construction du script de test
    # LiveCodeBench fournit généralement un dictionnaire 'inputs' et 'outputs'
    # Nous créons un script de test local pour exécuter ces assertions.
    test_script = generated_code + "\n\n"
    
    try:
        io_data = json.loads(input_output)
        test_script += "import json\n"
        test_script += f"inputs = {io_data.get('inputs', [])}\n"
        test_script += f"expected_outputs = {io_data.get('outputs', [])}\n"
        test_script += f"fn_name = '{io_data.get('fn_name', '')}'\n"
        
        # Test runner statique ajouté à la volée
        test_script += """
if fn_name and fn_name in globals():
    func = globals()[fn_name]
    for i in range(len(inputs)):
        # Handle varargs vs normal args based on type
        inp = inputs[i]
        expected = expected_outputs[i]
        if isinstance(inp, list):
            res = func(*inp)
        else:
            res = func(inp)
            
        if res != expected:
            print(f"Test failed: input {inp}, expected {expected}, got {res}")
            import sys
            sys.exit(1)
    print("All tests passed!")
else:
    print("Function not found!")
    import sys
    sys.exit(1)
"""
    except Exception as e:
        # Fallback if json parsing fails
        return -1.0

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_filename = f.name
        
    try:
        result = subprocess.run(
            [sys.executable, temp_filename],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        if result.returncode == 0 and "All tests passed!" in result.stdout:
            reward = 1.0  
        else:
            reward = -1.0 
    except subprocess.TimeoutExpired:
        reward = -1.0 
    except Exception:
        reward = -1.0
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    return reward

def load_livecodebench_dataset():
    """
    Charge la version allégée/test de LiveCodeBench depuis HuggingFace.
    Nécessite la librairie 'datasets'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        sys.exit(1)
        
    print("Downloading dataset (LiveCodeBench)...")
    # On utilise la version de test pour avoir les problèmes récents (sans contamination)
    # Note: On limite au split test pour aller vite sur ce script
    ds = load_dataset("livecodebench/code_generation", split="test")
    print(f"Loaded {len(ds)} LiveCodeBench problems.")
    return ds

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
    
    dataset = load_livecodebench_dataset()
    
    print("\nStarting REAL Training Loop with LiveCodeBench...")
    metrics_history = []
    
    for step, data in enumerate(dataset):
        prompt_text = data["question_content"]
        input_output_str = data["input_output"] # JSON string des I/O
        
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
                max_new_tokens=300, # Vraie génération de code
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.2, # Low temp pour le code
                do_sample=True
            )
            generated_ids = gen_outputs[0][inputs_embeds.shape[1] - 1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        # Evaluate using real subprocess Sandbox against LiveCodeBench I/O
        reward = execute_lcb_reward(generated_text, input_output_str)
        metrics_history.append(1 if reward > 0 else 0)
        
        # Double Update 
        policy_loss = -reward * soft_token.norm() 
        total_loss = policy_loss + 0.1 * sigreg_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(archi.parameters(), 1.0)
        optimizer.step()
        
        # Darwinian update on MoE
        archi.moe.distribute_reward(reward)
        
        if (step + 1) % 5 == 0:
            recent_acc = sum(metrics_history[-10:]) / min(10, len(metrics_history))
            print(f"Step {step+1:4d} | Recent Acc: {recent_acc*100:.1f}% | Reward: {reward:2.0f} | Loss: {total_loss.item():.4f}")

    print("\nReal Phase 5 Training Completed.")

if __name__ == '__main__':
    run_real_phase5()

