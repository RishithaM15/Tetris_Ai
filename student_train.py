from os import pardir
import numpy as np
from game import Game
from custom_model import CUSTOM_AI_MODEL
import random
import pandas as pd
import pickle

def cross(a1, a2):

    new_weights = []
    
    prop = a1.fit_rel / a2.fit_rel
    
    for i in range(len(a1.weights)):
        rand = random.uniform(0, 1)
        if rand > prop:
            new_weights.append(a1.weights[i])
        else:
            new_weights.append(a2.weights[i])

    return CUSTOM_AI_MODEL(weights = np.array(new_weights), mutate = True)

def compute_fitness(agent, trials):

    fitness = []
    
    for i in range(trials):
        game = Game('student', agent = agent)
        peices_dropped, rows_cleared = game.run_no_visual()
        fitness.append(peices_dropped)
     
    return np.average(np.array(fitness))

def run_X_epochs(num_epochs=10, num_trials=20, pop_size=100, num_elite=10, survival_rate=0.2):
    data = []
    population = [CUSTOM_AI_MODEL() for _ in range(pop_size)]

    for epoch in range(num_epochs):
        total_fitness = 0
        top_agent = None

        # Evaluate each agent
        for agent in population:
            agent.fit_score = compute_fitness(agent, trials=num_trials)
            total_fitness += agent.fit_score

        # Normalize fitness scores
        for agent in population:
            agent.fit_rel = agent.fit_score / total_fitness

        # Sort population by fitness
        sorted_pop = sorted(population, reverse=True)
        next_gen = []

        # Retain elite agents
        for i in range(num_elite):
            next_gen.append(CUSTOM_AI_MODEL(weights=sorted_pop[i].weights, mutate=False))

        # Select parents and create offspring
        num_parents = round(pop_size * survival_rate)
        parents = sorted_pop[:num_parents]
        for _ in range(pop_size - num_elite):
            parent1, parent2 = random.sample(parents, 2)
            next_gen.append(cross(parent1, parent2))

        # Replace population with next generation
        population = next_gen

        # Log best agent and weights
        top_agent = sorted_pop[0]
        data.append(top_agent.weights)
        print(f"Epoch {epoch + 1}: Best Fitness = {top_agent.fit_score}")

        # Save data after each epoch
        with open("data.txt", "wb") as f:
            pickle.dump(data, f)

    return data

run_X_epochs()
