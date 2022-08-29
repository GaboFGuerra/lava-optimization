# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from time import time

import numpy as np

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.processes import (
    ReadGate,
    SolutionReadout,
    CostConvergenceChecker,
)
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver


class TestOptimizationSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = OptimizationSolver()
        self.problem = QUBO(
            np.asarray(
                [[-5, 2, 4, 0], [2, -3, 1, 0], [4, 1, -8, 5], [0, 0, 5, -6]]
            )
        )
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)

    def test_create_obj(self):
        self.assertIsInstance(self.solver, OptimizationSolver)

    def test_solution_has_expected_shape(self):
        solution = self.solver.solve(self.problem, timeout=3000)
        self.assertEqual(solution.shape, self.solution.shape)

    def test_solve_method(self):
        np.random.seed(2)
        solution = self.solver.solve(self.problem, timeout=20)
        print(solution)
        self.assertTrue((solution == self.solution).all())

    @unittest.skip("WIP")
    def test_solve_map_coloring(self):
        # np.random.seed(2)/
        q = np.asarray(
            [
                [
                    -4.0,
                    4.0,
                    4.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                ],
                [
                    4.0,
                    -4.0,
                    4.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                ],
                [
                    4.0,
                    4.0,
                    -4.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                ],
                [
                    2.0,
                    0.0,
                    0.0,
                    -4.0,
                    4.0,
                    4.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    2.0,
                    0.0,
                    4.0,
                    -4.0,
                    4.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    4.0,
                    -4.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    -4.0,
                    4.0,
                    4.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    4.0,
                    -4.0,
                    4.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    4.0,
                    -4.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    -4.0,
                    4.0,
                    4.0,
                    2.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    4.0,
                    -4.0,
                    4.0,
                    0.0,
                    2.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    4.0,
                    -4.0,
                    0.0,
                    0.0,
                    2.0,
                ],
                [
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    -4.0,
                    4.0,
                    4.0,
                ],
                [
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    4.0,
                    -4.0,
                    4.0,
                ],
                [
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    4.0,
                    -4.0,
                ],
            ]
        )
        self.problem = QUBO(q)
        solution = self.solver.solve(self.problem, timeout=100000)
        cost = solution @ q @ solution
        reference_solution = np.zeros(15)
        np.put(reference_solution, [1, 3, 8, 10, 14], 1)
        expected_cost = reference_solution @ q @ reference_solution
        self.assertEqual(cost, expected_cost)

    def test_solver_creates_optimization_solver_process(self):
        solver_process = self.solver._create_solver_process(self.problem)
        class_name = type(solver_process).__name__
        self.assertIs(solver_process, self.solver._solver_process)
        self.assertEqual(class_name, "OptimizationSolverProcess")

    def test_solves_creates_macrostate_reader_processes(self):
        self.assertIsNone(self.solver._solver_process)
        self.solver.solve(self.problem, timeout=1)
        mr = self.solver._solver_process.model_class(
            self.solver._solver_process
        ).macrostate_reader
        self.assertIsInstance(mr.read_gate, ReadGate)
        self.assertIsInstance(mr.solution_readout, SolutionReadout)
        self.assertEqual(
            mr.solution_readout.solution.shape,
            (self.problem.variables.num_variables,),
        )
        self.assertIsInstance(mr.cost_convergence_check, CostConvergenceChecker)

    def test_macrostate_reader_processes_connections(self):
        self.assertIsNone(self.solver._solver_process)
        self.solver.solve(self.problem, timeout=1)
        mr = self.solver._solver_process.model_class(
            self.solver._solver_process
        ).macrostate_reader
        self.assertIs(
            mr.cost_convergence_check.s_out.out_connections[0].process,
            mr.read_gate,
        )
        self.assertIs(
            mr.read_gate.out_port.out_connections[0].process,
            mr.solution_readout,
        )
        self.assertIs(
            self.solver._solver_process.variable_assignment.aliased_var,
            mr.solution_readout.solution,
        )
        self.assertIs(
            self.solver._solver_process.variable_assignment.aliased_var.process,
            mr.solution_readout,
        )

    def test_cost_checker_is_connected_to_variables_population(self):
        self.assertIsNone(self.solver._solver_process)
        self.solver.solve(self.problem, timeout=1)
        pm = self.solver._solver_process.model_class(
            self.solver._solver_process
        )
        mr = pm.macrostate_reader
        self.assertIs(
            mr.cost_convergence_check.s_in.in_connections[0].process,
            pm.variables.discrete,
        )

    def test_qubo_cost_defines_weights(self):
        self.solver.solve(self.problem, timeout=1)
        pm = self.solver._solver_process.model_class(
            self.solver._solver_process
        )
        q_no_diag = np.copy(self.problem.cost.get_coefficient(2))
        np.fill_diagonal(q_no_diag, 0)
        condition = (
            pm.cost_minimizer.coefficients_2nd_order.weights.init == -q_no_diag
        ).all()
        self.assertTrue(condition)

    def test_qubo_cost_defines_biases(self):
        self.solver.solve(self.problem, timeout=1)
        pm = self.solver._solver_process.model_class(
            self.solver._solver_process
        )
        condition = (
            pm.variables.discrete.importances
            == -self.problem.cost.get_coefficient(2).diagonal()
        ).all()
        self.assertTrue(condition)

    def test_qubo_cost_defines_num_vars_in_discrete_variables_process(self):
        self.solver.solve(self.problem, timeout=1)
        pm = self.solver._solver_process.model_class(
            self.solver._solver_process
        )
        self.assertEqual(
            pm.variables.discrete.num_variables,
            self.problem.variables.num_variables,
        )
        self.assertEqual(
            self.solver._solver_process.variable_assignment.size,
            self.problem.variables.num_variables,
        )

    def test_solver_stops_when_solution_found(self):
        t_start = time()
        solution = self.solver.solve(self.problem, timeout=-1)
        t_end = time()
        self.assertTrue(t_start - t_end < 1)


if __name__ == "__main__":
    unittest.main()
