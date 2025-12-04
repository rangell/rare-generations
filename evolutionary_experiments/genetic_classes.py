import logging
from abc import ABC, abstractmethod

from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from copy import deepcopy
import numpy as np
import os
import json

class Particle:
    def __init__(self, *, ancestor=None, data=None, score=None, generation=None, info=None, **kwargs):
        self.ancestor = ancestor
        self.data = data
        self.score = score
        self.generation = generation
        self.info = info
        self.kwargs = kwargs

    def __str__(self):
        return f"Particle(ancestor=..., data={self.data}, score={self.score}, generation={self.generation}, info={self.info}, **{self.kwargs})"

    def to_dict(self):
        as_dict = self.__dict__
        as_dict["ancestor"] = str(as_dict["ancestor"])
        return as_dict


class Experiment(ABC):
    def __init__(self, init_population_size: int, mutations_per_particle: int, preserve_ancestor: bool = True):
        self.mutations_per_particle = mutations_per_particle
        self.init_population_size = init_population_size
        self.preserve_ancestor = preserve_ancestor
        logger.info(f"Initializing experiment with {init_population_size} particles")
        self.population = self.initialize_population(init_population_size)
        self.generation = 0
        logger.info(f"\tEvaluating {len(self.population)} particles")
        self.evaluate_particle_list(self.population)
        logger.info(f"...Done\n")

        
    @abstractmethod
    def initialize_population(self, init_population_size: int, **kwargs) -> list[Particle]:
        pass

    @abstractmethod
    def evaluate(self, particle: Particle) -> float:
        pass

    def mutate(self, particle: Particle) -> Particle:
        return particle

    @abstractmethod
    def select(self, particles: list[Particle]) -> list[Particle]:
        pass   

    def evaluate_particle_list(self, particles: list[Particle]):
        has_score = 0
        for particle in tqdm(particles, desc="Evaluating particles"):
            if particle.score is not None:
                has_score += 1
            else:
                particle.score = self.evaluate(particle)
        
        if has_score > 0:
            logger.info(f"{has_score}/{len(particles)} particles already evaluated, skipped")
       

    def mutate_particle_list(self, particles: list[Particle], mutations_per_particle: int, preserve_ancestor: bool = True) -> list[Particle]:
        mutated_particles = []
        for particle in particles:
            for i in range(mutations_per_particle):
                mutated_particle = deepcopy(particle)
                mutated_particle.generation = particle.generation + 1
                mutated_particle.ancestor = particle
                if preserve_ancestor and i == 0:
                    # keep the ancestor around
                    pass
                else:
                    mutated_particle.score = None
                    self.mutate(particle_for_mutation=mutated_particle)
                mutated_particles.append(mutated_particle)
        return mutated_particles

    def run_step(self):
        mutated_population = self.mutate_particle_list(self.population, self.mutations_per_particle, self.preserve_ancestor)
        self.evaluate_particle_list(mutated_population)
        self.population = self.select(mutated_population)
        self.generation += 1
    
    def get_population_stats(self):
        best_score = max(self.population, key=lambda x: x.score).score
        mean_score = np.mean([x.score for x in self.population])
        logger.info(f"Best score: {best_score}, Mean score: {mean_score}, N={len(self.population)}")
        return dict(best_score=best_score, mean_score=mean_score, N=len(self.population))

    def run(self, num_steps: int):
        logger.info(f"Running experiment for {num_steps} steps")
        pop_stats = [self.get_population_stats()]
        for step in tqdm(range(num_steps), desc="Running experiment"):
            self.run_step()
            pop_stats.append(self.get_population_stats())
        return self.population, pop_stats

    def write_population_data(self, out_path: str):
        with open(out_path, 'w') as f:
            json.dump([p.to_dict() for p in self.population], f, indent=4)
  
if __name__ == "__main__":
    class TestExperiment(Experiment):       
        def initialize_population(self, init_population_size: int):
            return [Particle(generation=0, info=i) for i in range(init_population_size)]

        def evaluate(self, particle: Particle):
            return particle.info

        def select(self, particles: list[Particle]):
            return sorted(particles, key=lambda x: x.score, reverse=True)[:self.init_population_size]

        def mutate(self, particle_for_mutation: Particle):
            particle_for_mutation.info += 1


    experiment = TestExperiment(init_population_size=10, mutations_per_particle=2)
    population, pop_stats = experiment.run(num_steps=5)
    experiment.write_population_data(out_path='test_population.json')
    assert pop_stats['N'] == 10
    assert pop_stats['best_score'] == 14
    assert pop_stats['mean_score'] == 14
    print(pop_stats)