import torch
import torch.nn as nn
import json
import tempfile
import subprocess
import sys
import os

# Ajout du chemin pour les imports locaux
sys.path.append(os.getcwd())
try:
    from phase5_real import ScrupulousArchitecture, execute_lcb_reward, execute_math_reward
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    sys.exit(1)

def test_architecture_tensors():
    print("--- Testing Architecture Tensors ---")
    batch_size = 1
    llm_hidden_dim = 4096
    latent_dim = 512
    num_experts = 3
    
    archi = ScrupulousArchitecture(
        llm_hidden_dim=llm_hidden_dim, 
        latent_dim=latent_dim, 
        num_experts=num_experts
    )
    
    # Mock des activations de la couche 9
    h_t_mock = torch.randn(batch_size, llm_hidden_dim)
    
    try:
        soft_token, sigreg_loss = archi(h_t_mock)
        print("✅ Forward pass successful")
        assert soft_token.shape == (batch_size, llm_hidden_dim), f"Expected {(batch_size, llm_hidden_dim)}, got {soft_token.shape}"
        print("✅ Soft token shape correct")
        assert isinstance(sigreg_loss.item(), float), "SIGReg loss is not a float"
        print("✅ SIGReg loss computed")
        
        # Test de l'attribution des rewards
        print("Testing Darwinian Reward Distribution...")
        # L'expert a été sélectionné pendant le forward pass
        archi.moe.distribute_reward(1.0)
        print("✅ Reward distributed successfully without backprop errors")
        
    except Exception as e:
        print(f"❌ Tensor test failed: {e}")
        return False
    return True

def test_sandbox_execution():
    print("\n--- Testing Code Sandbox Execution ---")
    
    # Mock LiveCodeBench format
    valid_code = "```python\ndef add(a, b):\n    return a + b\n```"
    invalid_code = "```python\ndef add(a, b):\n    return a - b\n```"
    timeout_code = "```python\ndef add(a, b):\n    while True: pass\n```"
    
    io_data = json.dumps({
        "fn_name": "add",
        "inputs": [[1, 2], [3, 4]],
        "outputs": [3, 7]
    })
    
    reward_valid = execute_lcb_reward(valid_code, io_data)
    reward_invalid = execute_lcb_reward(invalid_code, io_data)
    reward_timeout = execute_lcb_reward(timeout_code, io_data, timeout=1) # timeout réduit pour le test
    
    try:
        assert reward_valid == 1.0, f"Valid code should yield 1.0, got {reward_valid}"
        print("✅ Valid code passed")
        assert reward_invalid == -1.0, f"Invalid code should yield -1.0, got {reward_invalid}"
        print("✅ Invalid code caught")
        assert reward_timeout == -1.0, f"Timeout code should yield -1.0, got {reward_timeout}"
        print("✅ Infinite loop caught")
    except AssertionError as e:
        print(f"❌ Sandbox test failed: {e}")
        return False
    return True

def test_math_evaluation():
    print("\n--- Testing Math Evaluation Regex ---")
    
    target_str = "So the final answer is #### 42"
    
    valid_text_1 = "The answer is 42."
    valid_text_2 = "Let's calculate: 40 + 2 = 42"
    valid_text_3 = "The number of apples is 42.0"
    
    invalid_text_1 = "The answer is 43."
    invalid_text_2 = "I don't know, maybe 24?"
    
    try:
        assert execute_math_reward(valid_text_1, target_str) == 1.0
        assert execute_math_reward(valid_text_2, target_str) == 1.0
        assert execute_math_reward(valid_text_3, target_str) == 1.0
        print("✅ Valid math extractions passed")
        
        assert execute_math_reward(invalid_text_1, target_str) == -1.0
        assert execute_math_reward(invalid_text_2, target_str) == -1.0
        print("✅ Invalid math extractions caught")
    except AssertionError as e:
        print(f"❌ Math regex test failed: {e}")
        return False
    return True

if __name__ == "__main__":
    print("========================================")
    print("Starting Local Exhaustive Tests")
    print("========================================")
    
    t1 = test_architecture_tensors()
    t2 = test_sandbox_execution()
    t3 = test_math_evaluation()
    
    print("\n========================================")
    if t1 and t2 and t3:
        print("🎉 ALL TESTS PASSED. Ready for Pod deployment.")
    else:
        print("⚠️ SOME TESTS FAILED. Fix issues before deploying to A100.")
