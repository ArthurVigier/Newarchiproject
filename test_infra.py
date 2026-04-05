import sys
import torch
import importlib

def test_dependencies():
    deps = ["transformers", "accelerate", "datasets", "tiktoken", "sentencepiece"]
    missing = []
    for dep in deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} is installed")
        except ImportError:
            print(f"❌ {dep} is missing")
            missing.append(dep)
    return missing

def test_model_loading_config(model_name="THUDM/glm-4-9b-chat"):
    from transformers import AutoConfig
    print(f"\nTesting config loading for {model_name}...")
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Config loaded")
        
        # Check for the specific GLM-4 issue
        if not hasattr(config, 'max_length'):
            print("⚠️ Config missing 'max_length' (Common GLM-4 issue)")
            if hasattr(config, 'seq_length'):
                print(f"ℹ️ Found 'seq_length': {config.seq_length}")
                config.max_length = config.seq_length
                print("✅ Patched config.max_length")
            else:
                print("❌ Could not find alternative for max_length")
        else:
            print("✅ Config has 'max_length'")
            
    except Exception as e:
        print(f"❌ Failed to load config: {e}")

def check_gpu():
    print("\nChecking GPU status...")
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"ℹ️ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("❌ No GPU detected")

if __name__ == "__main__":
    print("--- Recurrent Infrastructure Test ---")
    missing_deps = test_dependencies()
    if missing_deps:
        print(f"\nACTION REQUIRED: pip install {' '.join(missing_deps)}")
    
    check_gpu()
    test_model_loading_config()
    print("\n--- Infrastructure Test Complete ---")
