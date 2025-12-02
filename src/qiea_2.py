#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qiea_2.pyy - Quantum Inspired Evolutionary Algorithm
author: Bill Thompson
license: GPL 3
copyright: 2025-11-29
"""
import numpy as np
import matplotlib.pyplot as plt
import time

class QIEAKnapsack:
    def __init__(self, 
                 weights: np.ndarray, 
                 values: np.ndarray, 
                 capacity: int, 
                 rng: np.random._generator.Generator,
                 population_size: int=50, 
                 max_generations: int=200) -> None:
        """
        Initialization

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
        population_size : int, optional
            size of evolutionary population by default 50
        max_generations : int, optional
            how many generations to run, by default 150
        """
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.pop_size = population_size
        self.max_gens = max_generations
        self.num_items = len(weights)
        self.rng = rng
        
        # Initialize quantum population with qubits [alpha, beta]
        # Each individual is a probability distribution over all items
        self.quantum_pop = np.ones((self.pop_size, self.num_items, 2)) * (1/np.sqrt(2))
        
        # Store best solution
        self.best_solution = self.quantum_pop[0].copy
        self.best_fitness = 0
        self.fitness_history = []
        
    def observe_quantum_individual(self, 
                                   quantum_individual: np.ndarray) -> np.ndarray:
        """
        Make a measurement of a quantum object

        Parameters
        ----------
        quantum_individual : np.ndarray
            a member of the quantum population

        Returns
        -------
        np.ndarray
            an observation, i.e. a set bits representing a solution
        """
        # Observe a quantum individual to generate a classical binary solution
        classical_solution = np.zeros(self.num_items, dtype=int)
        
        for i in range(self.num_items):
            alpha, beta = quantum_individual[i]
            prob_1 = beta**2  # Probability of observing 1
            if self.rng.random() < prob_1:
                classical_solution[i] = 1
                
        return classical_solution
    
    def calculate_fitness(self, 
                          solution: np.ndarray) -> int:
        """
        Calculate fitness of solution
        Apply penalty if capacity is exceeded

        Parameters
        ----------
        solution : np.ndarray
            a classical individual

        Returns
        -------
        int
            the fitness value
        """
        total_weight = np.dot(solution, self.weights)
        total_value = np.dot(solution, self.values)
        
        # Penalty for exceeding capacity
        if total_weight > self.capacity:
            # Excess ratio penalty
            excess_ratio = (total_weight - self.capacity) / self.capacity
            penalty = 0.5 * total_value * excess_ratio
            total_value -= penalty
            
        return max(total_value, 0)  # Ensure non-negative fitness
    
    def repair_solution(self, 
                        solution: np.ndarray) -> np.ndarray:
        """
        If solution is over capacity until it is valid

        Parameters
        ----------
        solution : np.ndarray
            a classical individuals

        Returns
        -------
        np.ndarray
            a valid solution
        """
        # pair infeasible solution using greedy approach
        total_weight = np.dot(solution, self.weights)
        
        if total_weight <= self.capacity:
            return solution
            
        # Create list of included items with their value-to-weight ratio
        included_items = []
        for i in range(self.num_items):
            if solution[i] == 1:
                ratio = self.values[i] / self.weights[i]
                included_items.append((i, ratio, self.weights[i]))
        
        # Sort by value-to-weight ratio (ascending - remove worst first)
        included_items.sort(key=lambda x: x[1])
        
        # Remove items until capacity constraint is satisfied
        repaired_solution = solution.copy()
        current_weight = total_weight
        
        for item_idx, ratio, weight in included_items:
            if current_weight > self.capacity:
                repaired_solution[item_idx] = 0
                current_weight -= weight
            else:
                break
                
        return repaired_solution
    
    def update_quantum_individual(self, 
                                  quantum_individual: np.ndarray, 
                                  best_classical: np.ndarray, 
                                  theta: float=0.05*np.pi) -> np.ndarray:
        """
        Update an quantum individual, rotate individual in drection of clalsical solution

        Parameters
        ----------
        quantum_individual : np.ndarray
            an element of quantum populations
        best_classical : np.ndarray
            the most fit individual
        theta : float, optional
            rotation value, by default 0.05*np.pi

        Returns
        -------
        np.ndarray
            updated quantum individual
        """
        # update quantum individual using rotation gate
        updated_individual = quantum_individual.copy()
        
        for i in range(self.num_items):
            alpha, beta = quantum_individual[i]
            
            # Determine rotation direction based on best solution
            if best_classical[i] == 1:
                # Rotate towards |1⟩ state
                new_alpha = alpha * np.cos(theta) - beta * np.sin(theta)
                new_beta = alpha * np.sin(theta) + beta * np.cos(theta)
            else:
                # Rotate towards |0⟩ state  
                new_alpha = alpha * np.cos(theta) + beta * np.sin(theta)
                new_beta = -alpha * np.sin(theta) + beta * np.cos(theta)
            
            # Normalize (should already be normalized due to rotation, but for numerical stability)
            norm = np.sqrt(new_alpha**2 + new_beta**2)
            updated_individual[i] = [new_alpha/norm, new_beta/norm]
            
        return updated_individual
    
    def mutate_quantum_individual(self, 
                                  quantum_individual: np.ndarray, 
                                  mutation_rate: float=0.01) -> np.ndarray:
        """
        Mutate an indivdual solution by flipping bits

        Parameters
        ----------
        quantum_individual : np.ndarray
            an element of quantum populations
            
        mutation_rate : float, optional
            mutation value, by default 0.01

        Returns
        -------
        np.ndarray
            mutated individual
        """
        # Apply mutation to quantum individual by randomly flipping qubit states
        mutated = quantum_individual.copy()
        
        for i in range(self.num_items):
            if self.rng.random() < mutation_rate:
                # Swap alpha and beta (equivalent to bit flip in classical space)
                alpha, beta = mutated[i]
                mutated[i] = [beta, alpha]
                
        return mutated
    
    def run(self) -> tuple[np.ndarray, int]:
        """
        Execute the QIEA algorithm
        
        Returns
        -------
        tuple[np.ndarray, int]
            best classical solution, fitness of best solution
        """
        # Execute the QIEA algorithm
        print("Starting Quantum-Inspired Evolutionary Algorithm for 0/1 Knapsack")
        print(f"Items: {self.num_items}, Capacity: {self.capacity}")
        print("-" * 60)
        
        for generation in range(self.max_gens):
            # Step 1: Observation - Generate classical population from quantum population
            classical_pop = []
            for i in range(self.pop_size):
                classical_sol = self.observe_quantum_individual(self.quantum_pop[i])
                classical_sol = self.repair_solution(classical_sol)
                classical_pop.append(classical_sol)
            
            # Step 2: Evaluation
            fitness_scores = [self.calculate_fitness(sol) for sol in classical_pop]
            
            # Find best solution in current generation
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_solution = classical_pop[current_best_idx]
            
            # Update global best
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution.copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Step 3: Update quantum population
            for i in range(self.pop_size):
                # Update towards best solution with some probability
                if self.rng.random() < 0.8:  # Learning rate
                    self.quantum_pop[i] = self.update_quantum_individual(
                        self.quantum_pop[i], self.best_solution
                    )
                
                # Apply mutation
                self.quantum_pop[i] = self.mutate_quantum_individual(self.quantum_pop[i])
            
            # Progress reporting
            if generation % 20 == 0 or generation == self.max_gens - 1:
                weight = np.dot(self.best_solution, self.weights)
                value = np.dot(self.best_solution, self.values)
                print(f"Generation {generation:3d}: Best Value = {value:.1f}, "
                      f"Weight = {weight:.1f}/{self.capacity}")
        
        return self.best_solution, self.best_fitness
    
    def plot_progress(self) -> None:
        """
        Plot the fitness progression over generations
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, linewidth=2)
        plt.title('QIEA Progress for 0/1 Knapsack Problem')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (Value)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_solution(self) -> None:
        """
        Print the final solution details
        """
        if self.best_solution is None:
            print("No solution found!")
            return
            
        total_weight = np.dot(self.best_solution, self.weights)
        total_value = np.dot(self.best_solution, self.values)
        
        print("\n" + "="*60)
        print("QIEA SOLUTION")
        print("="*60)
        print(f"Total Value: {total_value:.2f}")
        print(f"Total Weight: {total_weight:.2f} / {self.capacity}")
        print(f"Capacity Utilization: {total_weight/self.capacity*100:.1f}%")
        
        selected_items = [i for i in range(self.num_items) if self.best_solution[i] == 1]
        print(f"Selected Items: {len(selected_items)}/{self.num_items}")
        
        print("\nSelected Items Details:")
        for i in selected_items:
            print(f"  Item {i+1}: Weight={self.weights[i]}, Value={self.values[i]}, "
                  f"Ratio={self.values[i]/self.weights[i]:.2f}")

# Example usage and test function
def test_knapsack() -> QIEAKnapsack:
    """
    Run the QIEA algorithm

    Returns
    -------
    QIEAKnapsack
        a QIEAKnapsack object
    """
    
    # # Sample problem from literature
    # weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    # values = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    # capacity = 165
    
    seed = 42
    # seed = int(time.time())   # uncomment to change to more random seed
    rng = np.random.default_rng(seed)
    num_items = 50
    weights = rng.integers(1, 50, num_items)
    values = rng.integers(10, 100, num_items)
    capacity = int(0.6 * np.sum(weights))
    
    # Expected optimal solution for this problem (known benchmark)
    # Optimal value: 309
    
    # Initialize and run QIEA
    qiea = QIEAKnapsack(weights, values, capacity, rng, 
                        population_size=30, max_generations=100)
    best_solution, best_fitness = qiea.run()
    
    return qiea

def main():
    # Run the basic test
    print("Testing QIEA on 0/1 Knapsack Problem")
    print("="*50)
    qiea = test_knapsack()
    
    # Display results
    qiea.print_solution()
    qiea.plot_progress()

if __name__ == "__main__":
    main()