import numpy as np

class WhaleOptimizationAlgorithm:
    def __init__(self, fitness_func, n_whales, n_iter, lower_bound, upper_bound):
        self.fitness_func = fitness_func
        self.n_whales = n_whales
        self.n_iter = n_iter
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.dim = len(lower_bound)
        
        # Initialize whale positions
        self.whale_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_whales, self.dim))
        
        # Evaluate initial fitness
        self.fitness_scores = np.array([self.fitness_func(p) for p in self.whale_positions])
        
        best_idx = np.argmin(self.fitness_scores)
        self.best_whale = self.whale_positions[best_idx].copy()
        self.best_score = self.fitness_scores[best_idx]
        
        self.history = []

    def optimize(self):
        for t in range(self.n_iter):
            a = 2 * (1 - t / self.n_iter)
            
            for i in range(self.n_whales):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                A = 2 * a * r1 - a
                C = 2 * r2
                
                if np.random.rand() < 0.5:
                    if abs(A) < 1:
                        D = abs(C * self.best_whale - self.whale_positions[i])
                        new_pos = self.best_whale - A * D
                    else:
                        rand_idx = np.random.randint(self.n_whales)
                        X_rand = self.whale_positions[rand_idx]
                        D = abs(C * X_rand - self.whale_positions[i])
                        new_pos = X_rand - A * D
                else:
                    distance_to_best = abs(self.best_whale - self.whale_positions[i])
                    # Spiral update
                    L = np.random.uniform(-1, 1)
                    new_pos = distance_to_best * np.exp(L) * np.cos(L * 2 * np.pi) + self.best_whale
                    
                self.whale_positions[i] = np.clip(new_pos, self.lower_bound, self.upper_bound)
                
            # Evaluate fitness of new positions
            self.fitness_scores = np.array([self.fitness_func(p) for p in self.whale_positions])
            best_idx = np.argmin(self.fitness_scores)
            
            # Key fix: Do not recalculate fitness(self.best_whale). Just compare to self.best_score.
            if self.fitness_scores[best_idx] < self.best_score:
                self.best_whale = self.whale_positions[best_idx].copy()
                self.best_score = self.fitness_scores[best_idx]
                
            self.history.append(float(self.best_score))
            print(f"Iteration {t+1}/{self.n_iter}, Best Fitness: {self.best_score:.4f}")
            
        return self.best_whale, self.best_score, self.history

