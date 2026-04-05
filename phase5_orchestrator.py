import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

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

def mock_reward_function(generated_code: str, expected_output: str) -> float:
    if expected_output.lower() in generated_code.lower():
        return 1.0
    return -1.0

def run_phase5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 5 on {device.type.upper()}...")
    
    # Qwen2.5-Coder-7B
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    print(f"Loading Base LLM: {model_name} (Frozen)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load frozen LLM
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    llm_model.eval()
    for param in llm_model.parameters():
        param.requires_grad = False
        
    llm_hidden_dim = llm_model.config.hidden_size
    print(f"LLM Hidden Dimension detected: {llm_hidden_dim}")

    print("Loading Experimental Architecture...")
    archi = FullArchitecture(llm_hidden_dim=llm_hidden_dim, latent_dim=512, num_experts=3)
    archi = archi.to(device)
    
    # Optimizer targets ONLY the differentiable parts
    optimizer = optim.AdamW([p for p in archi.parameters() if p.requires_grad], lr=5e-5)
    
    LAYER_IDX = 9 
    
    mock_dataset = [
        {"prompt": "Write a python function to add two numbers.", "expected": "return"},
        {"prompt": "Write a python function to multiply two numbers.", "expected": "*"},
        {"prompt": "Write a python function that returns True.", "expected": "True"}
    ] * 10
    
    print("\nStarting Training Loop...")
    for step, data in enumerate(mock_dataset):
        prompt = data["prompt"]
        expected = data["expected"]
        
        optimizer.zero_grad()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = llm_model(**inputs, output_hidden_states=True)
            # Extract activations
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
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            generated_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            
        # Evaluate
        reward = mock_reward_function(generated_text, expected)
        
        # Double Update 
        # 1. Backprop on Projector/Encoder
        policy_loss = -reward * soft_token.norm() 
        total_loss = policy_loss + 0.1 * sigreg_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(archi.parameters(), 1.0)
        optimizer.step()
        
        # 2. Darwinian update on MoE
        archi.moe.distribute_reward(reward)
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1:2d}/30 | Reward: {reward:2.0f} | Loss: {total_loss.item():.4f}")

    print("\nPhase 5 Orchestrator test completed successfully!")

if __name__ == '__main__':
    run_phase5()
