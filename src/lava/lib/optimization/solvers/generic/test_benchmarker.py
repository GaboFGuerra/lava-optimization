# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest
from lava.lib.optimization.utils.solver_benchmarker import SolverBenchmarker 


class TestSolverBenchmarker (unittest.TestCase):
    
        
    def setUp(self) -> None:
        self.benchmarker= SolverBenchmarker()

    def test_benchmarker_instance_is_created(self):
        self.assertIsInstance(self.benchmarker, SolverBenchmarker)

    def test_setup_benchmark_call(self):
        self.benchmarker.setup_benchmark()
        
