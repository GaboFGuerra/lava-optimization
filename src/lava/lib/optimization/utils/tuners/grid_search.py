# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver


class SolverTuner:
    def __init__(self,
                 step_range=[1],
                 noise_range=[4],
                 steps_to_fire_range=[8]
                 ):
        self._step_range = step_range
        self._noise_range = noise_range
        self._steps_to_fire_range = steps_to_fire_range

    def tune(self,
             solver: OptimizationSolver,
             solver_parameters: ty.Dict,
             target_cost: int):
        hyperparameters = self._perform_grid_search(solver,
                                                    solver_parameters,
                                                    target_cost=target_cost)
        succeeded = hyperparameters is not None
        if hyperparameters:
            solver.hyperparameters =  hyperparameters
        return solver, succeeded

    def _perform_grid_search(self,
                             solver,
                             solver_parameters,
                             target_cost=None,
                             ) -> ty.Union[ty.NoReturn, ty.Dict]:
        qubo_problem = solver.problem
        for steps_to_fire in self._step_range:
            for noise_amplitude in self._noise_range:
                for step_size in self._steps_to_fire_range:
                    hyperparameters = dict(steps_to_fire=steps_to_fire,
                                           noise_amplitude=noise_amplitude,
                                           step_size=step_size
                                           )
                    print(f"{hyperparameters=}")
                    solver_parameters["hyperparameters"] = hyperparameters
                    print(f"{solver_parameters=}")
                    solution = solver.solve(**solver_parameters)
                    cost = qubo_problem.compute_cost(state_vector=solution)
                    self.print_cost_msg(cost, target_cost, solution)
                    if cost <= target_cost:
                        self.print_solution_found_msg(hyperparameters)
                        break
                else:
                    continue
                break
            else:
                continue
            break
        if cost > target_cost:
            hyperparameters = None
        return hyperparameters

    def print_cost_msg(self, cost, cost_ref, solution):
        msg = f"""Solution vector from Loihi {solution} \n
                      Nodes in maximum independent set (index starts at 0):
                      {np.where(solution)[0]}\n
                      QUBO cost of solution: {cost} (Lava) vs {cost_ref,} 
                      (Networkx)\n"""
        print(msg)

    def print_solution_found_msg(self, hyperparameters):
        print("Solution found!")
        print(f"{hyperparameters=}")

