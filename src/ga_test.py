import numpy as np
import time

    # Simple Traditional GA for comparison
class SimpleGA:
    def __init__(self, 
                 weights: np.ndarray, 
                 values: np.ndarray, 
                 capacity: int, 
                 rng: np.random._generator.Generator,
                 pop_size: int=50, 
                 max_gens: int=150) -> None:
        """
        Initialize GA

        Parameters
        ----------
        weights : np.ndarray
            weight of each object
        values : np.ndarray
            value of each object
        capacity : int
            how much weight can knapsack handle
        rng : np.random._generator.Generator
            random number generator
        pop_size : int, optional
            size of evolutionary population by default 50
        max_gens : int, optional
            how many generations to run, by default 150
        """
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.num_items = len(weights)
        self.rng = rng
        
    def run(self) -> tuple[np.ndarray, float]:
        """
        Execute the GA

        Returns
        -------
        tuple[np.ndarray, float]
            the population element with the best solution, the fitnss of the best solution
        """
        print("Starting Genetic Algorithm for 0/1 Knapsack")
        print(f"Items: {self.num_items}, Capacity: {self.capacity}")
        print("-" * 60)
        
        # Initialize population
        pop = self.rng.integers(0, 2, (self.pop_size, self.num_items))
        
        best_solution = pop[0].copy()
        best_fitness = 0
        
        for gen in range(self.max_gens):
            # Evaluate
            fitness = []
            for sol in pop:
                weight = np.dot(sol, self.weights)
                value = np.dot(sol, self.values)
                if weight > self.capacity:
                    value *= 0.5  # Penalty
                fitness.append(value)
            
            # Find best
            idx = np.argmax(fitness)
            if fitness[idx] > best_fitness:
                best_fitness = fitness[idx]
                best_solution = pop[idx].copy()
            
            # Selection and crossover
            new_pop = []
            for _ in range(self.pop_size):
                # Tournament selection
                p1, p2 = self.rng.choice(self.pop_size, 2, replace=False)
                parent1, parent2 = pop[p1], pop[p2]
                
                # Crossover
                crossover_point = self.rng.integers(1, self.num_items-1)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                
                # Mutation
                for i in range(self.num_items):
                    if self.rng.random() < 0.01:
                        child[i] = 1 - child[i]
                
                new_pop.append(child)
            
            pop = np.array(new_pop)
        
        return best_solution, best_fitness
    
def main() -> None:
    seed = 42  
    # seed = int(time.time())   # uncomment to change to more random seed
    rng = np.random.default_rng(seed)
    num_items = 50
    weights = rng.integers(1, 50, num_items)
    values = rng.integers(10, 100, num_items)
    capacity = int(0.6 * np.sum(weights))
    
    # QIEA parameters
    population_size = 20
    num_generations = 100
    rotation_angle_step = 0.01 * np.pi # A small step, e.g., 0.01 * pi radians

    # Run traditional GA
    ga = SimpleGA(weights, values, capacity, rng)
    ga_solution, ga_fitness = ga.run()
    ga_value = np.dot(ga_solution, values)
    ga_weight = np.dot(ga_solution, weights)
     
    print("\n" + "="*60)
    print("GA SOLUTION")
    print("="*60)
    print(f"Total Value: {ga_value:.2f}")
    print(f"Total Weight: {ga_weight:.2f} / {capacity}")
    print(f"Capacity Utilization: {ga_weight/capacity*100:.1f}%")

if __name__ == "__main__":
    main()