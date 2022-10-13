# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
# See: https://spdx.org/licenses/
# SPDX-License-Identifier: BSD-3-Clause

from lava.utils.system import Loihi2
from tests.lava.test_utils.utils import Utils
import logging
import numpy as np
import matplotlib.pyplot as plt
from lava.magma.compiler.subcompilers.constants import \
    MAX_EMBEDDED_CORES_PER_CHIP
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps

def is_loihi2_available(self):
    runtime_test =Utils.is_loihi2_available \
    and Utils.get_bool_env_setting("RUN_LOIHI_TESTS")
    Loihi2.preferred_partition = 'kp_build' #selelct preferred partition
    return Loihi2.is_loihi2_available
    
def import_profiler_if_loihi2_is_available(self):
    if self.is_loihi2_available():
        print(f'Running on {Loihi2.partition}')
        from lava.utils import loihi2_profiler
    else:
        RuntimeError("Loihi2 compiler is not available in this system. "
                "Problem benchmarking cannot proceed.")

import_profiler_if_loihi2_is_available()


class SolverBenchmarker():
    """Measure power and execution time for an optimization solver."""
    def __init__(self, num_steps: int):
        self._power_logger = None
        self._time_logger = None
        self.num_steps = num_steps
                       
    def setup_power_measurement(self, board):
        #configures profiling tools
        self._power_logger = loihi2_profiler.Loihi2Power(num_steps=self.num_steps)
        """The profiler tools can be enabled on the Loihi 2 system 
        as the workload runs through pre_run_fxs and post_run_fxs
        which are used to attach the profiling tools."""
        pre_run_fxs = [
            lambda board: self._power_logger.attach(board),
        ]
        post_run_fxs = [
            lambda board: self._power_logger.get_results(),
        ]
        run_config = Loihi2HwCfg(pre_run_fxs=pre_run_fxs,
                         post_run_fxs=post_run_fxs)
        self._log_config.level = logging.INFO
        self.run(condition=RunSteps(num_steps=self.num_steps), run_cfg=run_config)
        self.stop()

        # post processing
        time_stamp = self._power_logger.time_stamp
        vdd_p = self._power_logger.vdd_power  # neurocore power
        vddm_p = self._power_logger.vddm_power  # memory power
        vddio_p = self._power_logger.vddio_power  # IO power
        total_power = self._power_logger.total_power  # Total power

        total_power_mean = np.mean(total_power)
        vdd_p_mean = np.mean(vdd_p)
        vddm_p_mean = np.mean(vddm_p)
        vddio_p_mean = np.mean(vddio_p)
        # measure static power
        static_total_power = self._power_logger.static_total_power
        print(f'Total Power   : {total_power_mean:.6f} W')
        print(f'Dynamic Power : '
                  f'{total_power_mean - static_total_power:.6f} W')
        print(f'Static Power  : {static_total_power:.6f} W')
        print(f'VDD Power     : {vdd_p_mean:.6f} W')
        print(f'VDD-M Power   : {vddm_p_mean:.6f} W')
        print(f'VDD-IO Power  : {vddio_p_mean:.6f} W')

        fig, ax = plt.subplots()
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(total_power, color=color, label='Total Power')
        ax.plot(np.zeros_like(total_power)
                      + self._power_logger.static_total_power,
                        linestyle='--', color=color, label='Total Static Power')
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(vdd_p, color=color, label='VDD Power')
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(vddm_p, color=color, label='VDD-M Power')
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(vddio_p, color=color, label='VDD-IO Power')
        ax.set_ylabel('Power (W)')
        ax.set_xticks([])
        ax.legend()
        plt.show()

    def setup_time_measurement(self, board):
        self._time_logger = loihi2_profiler.Loihi2ExecutionTime()
        pre_run_fxs = [
            lambda board: self._time_logger.attach(board),
        ]
        post_run_fxs = [
            lambda board: self._time_logger.get_results(),
        ]

        run_config = Loihi2HwCfg(pre_run_fxs=pre_run_fxs,
                         post_run_fxs=post_run_fxs)
        self._log_config.level = logging.INFO
        self.run(condition=RunSteps(num_steps=self.num_steps), run_cfg=run_config)
        self.stop()
        plt.plot(time_series, label='Network Runtime')
        plt.ylabel('Time per timestep (us)')
        plt.xlabel('timestep')
        plt.show()