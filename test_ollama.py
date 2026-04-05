import subprocess
import json

def get_ollama_embeddings(prompt, model="llama3.1:8b-instruct-q6_K"):
    # Utilisation de l'API REST d'Ollama (souvent sur le port 11434)
    import urllib.request
    import urllib.parse
    
    url = "http://localhost:11434/api/embeddings"
    data = json.dumps({
        "model": model,
        "prompt": prompt
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("embedding", [])
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

if __name__ == "__main__":
    emb = get_ollama_embeddings("print('Hello World')")
    if emb:
        print(f"Successfully got embeddings. Length: {len(emb)}")
    else:
        print("Failed to get embeddings.")
