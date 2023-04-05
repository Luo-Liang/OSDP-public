import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from typing import List, Optional

from osdp_performance_model import ExecutionEnvironment, OSDPExecutionSimulator, dummy_compute_perf_model, sim, exec_env

from hyperopt import fmin, tpe
from hyperopt import hp


class OptimallyShardedDataParallelOrchestrator:
    def __init__(self, sim: OSDPExecutionSimulator, total_memory: float, exec_env: ExecutionEnvironment) -> None:
        # use Bayesian optimizer
        self.simulator = sim
        self.memory = total_memory
        # create search space
        # create optimizer
        self.exec_env = exec_env
        

    def optimize(self):
        all_choices = [ss for ss in ShardingStrategy]
        # print(f"all choices {all_choices}, types = {[x.value for x in ShardingStrategy]}")
        space = {}
        for desc in self.simulator.compute_perf_model:
            space[desc.name] = hp.choice(desc.name, [ss for ss in ShardingStrategy])    

        def objective(choices):
            for desc in self.simulator.compute_perf_model:
                desc.sharding_strategy = choices[desc.name]

            lat, mem = self.simulator.simulate(
                self.exec_env
            )

            if self.memory > mem:
                return 99999999999
            else:
                return lat

        output = fmin(objective, space, tpe.suggest, max_evals=100) # TPE good for discrete variables
        # print(f"raw output = {output}")
        for key in list(output.keys()):
            val = output[key]
            output[key] = [x for x in all_choices if x.value == val][0]

        return output
        



osdp = OptimallyShardedDataParallelOrchestrator(sim, 100, exec_env)
print(osdp.optimize())