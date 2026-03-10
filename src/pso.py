import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, fitness_func, n_particles, n_iter, lower_bound, upper_bound, w=0.5, c1=1.5, c2=1.5):
        self.fitness_func = fitness_func
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.dim = len(lower_bound)
        
        # PSO hyperparameters
        self.w = w    # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        
        # Initialize particles
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        self.velocities = np.zeros((self.n_particles, self.dim))
        
        # Initialize personal bests
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([self.fitness_func(p) for p in self.positions])
        
        # Initialize global best
        best_idx = np.argmin(self.pbest_scores)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_score = self.pbest_scores[best_idx]
        
        self.history = []

    def optimize(self):
        for t in range(self.n_iter):
            for i in range(self.n_particles):
                # Random coefficients
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Update velocity
                cognitive_velocity = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity
                
                # Update position
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # Apply boundary constraints
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)
                
                # Evaluate fitness
                current_score = self.fitness_func(self.positions[i])
                
                # Update personal best
                if current_score < self.pbest_scores[i]:
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_scores[i] = current_score
                    
                    # Update global best
                    if current_score < self.gbest_score:
                        self.gbest_position = self.positions[i].copy()
                        self.gbest_score = current_score
                        
            self.history.append(float(self.gbest_score))
            print(f"Iteration {t+1}/{self.n_iter}, Best Fitness: {self.gbest_score:.4f}")
            
        return self.gbest_position, self.gbest_score, self.history
