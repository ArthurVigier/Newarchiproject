import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_auc_experiment():
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)
    
    # Layers to test
    layers_to_test = [9, 12, 16]
    
    # Dataset un peu plus varié
    pass_code = [
        "def add(a, b): return a + b", "def check(x): return x > 0", "def f(s): return s.strip()",
        "def get(d, k): return d.get(k)", "def p(x): print(x)", "def sq(x): return x*x",
        "def identity(x): return x", "def is_none(x): return x is None", "def double(x): return x*2",
        "def power(a,b): return a**b", "def sub(a,b): return a-b", "def div(a,b): return a/b if b!=0 else 0",
        "def tail(l): return l[1:]", "def head(l): return l[0]", "def empty(l): return len(l) == 0",
        "def const(): return 42", "def neg(x): return -x", "def inv(b): return not b",
        "def join(a,b): return a + b", "def find(s,c): return c in s"
    ]
    fail_code = [
        "def add(a, b): return a + ", "def check(x): return x > ", "def f(s): return s.strip(",
        "def get(d, k): d.get(k", "def p(x): print(x", "def sq(x): x*x",
        "def identity(x): return", "def is_none(x): x is", "def double(x): x *",
        "def power(a,b): a**", "def sub(a,b): -b", "def div(a,b): return /b",
        "def tail(l): l[1:", "def head(l): l[0", "def empty(l): return len(l) =",
        "def const(): return", "def neg(x): return -", "def inv(b): not b",
        "def join(a,b): a +", "def find(s,c): c in"
    ]
    
    prompts = pass_code + fail_code
    labels = [1]*len(pass_code) + [0]*len(fail_code)
    
    all_activations = {layer: [] for layer in layers_to_test}
    
    print(f"Extracting activations for layers {layers_to_test}...")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for layer_idx in layers_to_test:
                # On prend l'état caché de la couche demandée, dernier token
                h = outputs.hidden_states[layer_idx][0, -1, :].numpy()
                all_activations[layer_idx].append(h)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(prompts)} examples.")
            
    for layer_idx in layers_to_test:
        X = np.array(all_activations[layer_idx])
        y = np.array(labels)
        
        # Scaling is important for LogisticRegression with strong regularization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/Test split avec shuffle
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        # Stronger regularization (C small) to handle low N/D ratio
        clf = LogisticRegression(C=0.1, penalty='l2', max_iter=1000)
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        
        print(f"Layer {layer_idx:2d} AUC: {auc:.4f}")
        if auc > 0.65:
            print(f"  --> LAYER {layer_idx} PASSED!")

if __name__ == '__main__':
    run_auc_experiment()
