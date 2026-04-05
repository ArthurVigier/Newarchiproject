import torch
import random
import itertools

class SurvivalPopulation:
    def __init__(self, n_neurons=50, input_dim=2, dormancy_threshold=15, death_threshold=50, rebirth_noise=1.0):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.dormancy_threshold = dormancy_threshold
        self.death_threshold = death_threshold
        self.rebirth_noise = rebirth_noise
        
        self.w_in = torch.randn(n_neurons, input_dim)
        self.b_in = torch.randn(n_neurons)
        self.w_out = torch.randn(n_neurons)
        
        self.scores = torch.zeros(n_neurons)
        self.dormancy = torch.zeros(n_neurons)
        
    def forward(self, x):
        h = torch.matmul(x, self.w_in.T) + self.b_in
        is_active = (h > 0)
        is_active = is_active & (self.dormancy <= self.dormancy_threshold)
        
        out = torch.sum(self.w_out[is_active])
        pred = 1.0 if out > 0 else 0.0
        return pred, is_active

    def update(self, is_active, reward):
        active_mask = is_active.bool()
        silent_mask = ~active_mask
        
        if reward > 0:
            self.scores[active_mask] += 1
            self.dormancy[active_mask] = 0
        else:
            self.dormancy[active_mask] += 2
            
        self.dormancy[silent_mask] += 1
        
        dead_mask = self.dormancy > self.death_threshold
        dead_indices = torch.where(dead_mask)[0]
        
        for i in dead_indices:
            self.rebirth(i)
            
    def rebirth(self, i):
        self.w_in[i] = torch.randn(self.input_dim) * self.rebirth_noise
        self.b_in[i] = torch.randn(1).item() * self.rebirth_noise
        self.w_out[i] = torch.randn(1).item() * self.rebirth_noise
        self.dormancy[i] = 0
        self.scores[i] = 0

def evaluate(dormancy_threshold, death_threshold, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    pop = SurvivalPopulation(n_neurons=50, input_dim=2, 
                             dormancy_threshold=dormancy_threshold, 
                             death_threshold=death_threshold)
    
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    
    steps = 1000
    window = []
    
    for step in range(1, steps + 1):
        idx = random.randint(0, 3)
        x, y_true = X[idx], Y[idx]
        
        pred, is_active = pop.forward(x)
        correct = (pred == y_true.item())
        window.append(1.0 if correct else 0.0)
        if len(window) > 100:
            window.pop(0)
            
        reward = 1.0 if correct else -1.0
        pop.update(is_active, reward)
        
    return sum(window) / len(window)

if __name__ == '__main__':
    dormancies = [5, 10, 15, 20, 30]
    deaths = [20, 30, 50, 80, 100, 200]
    
    best_acc = 0
    best_params = None
    
    for d, dt in itertools.product(dormancies, deaths):
        if dt <= d:
            continue
        accs = [evaluate(d, dt, seed=s) for s in [42, 43, 44]]
        mean_acc = sum(accs) / len(accs)
        print(f"Dormancy: {d:2d}, Death: {dt:3d} | Mean Acc: {mean_acc:.2f}")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = (d, dt)
            
    print(f"\nBest Params: Dormancy {best_params[0]}, Death {best_params[1]} with Acc: {best_acc:.2f}")
