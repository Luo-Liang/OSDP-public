from collections import deque
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple
import pandas
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.distributed.fsdp.api import ShardingStrategy

# HW_SKU, COLLECTIVE_NAME, BUFFER_SIZE, LATENCY
# SMC, REDUCE_SCATTER, 16384, scipy-tabular data
dtype2size = {
    "float": 4,
    "bfloat16": 2,
    "float16": 2
}

# feel free to overfit
class CommsPerformanceModel:
    @staticmethod
    def get_model_schema():
        return ["sku", "collectives", "world_size", "buffer_size", "param_dtype", "latency"]

    @staticmethod
    def get_categorical_features():
        return ["sku", "collectives", "param_dtype"]

    def preproc(self, data_frame: pandas.DataFrame, fit=False):
        onehotcolumns = data_frame[CommsPerformanceModel.get_categorical_features(
        )]
        if fit:
            transformed_columns = pandas.DataFrame(
                self.one_hot_encoder.fit_transform(onehotcolumns))
        else:
            transformed_columns = pandas.DataFrame(
                self.one_hot_encoder.transform(onehotcolumns))
        transformed_columns.columns = self.one_hot_encoder.get_feature_names_out(
            CommsPerformanceModel.get_categorical_features())
        numerical_columns = data_frame.drop(
            CommsPerformanceModel.get_categorical_features(), axis=1)
        return pandas.concat([transformed_columns, numerical_columns], axis=1)

    def __init__(self, data_frame: pandas.DataFrame) -> None:
        self.learner = GradientBoostingRegressor()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        label_name = "latency"
        lat_frame = data_frame[label_name]
        train_frame = self.preproc(data_frame.drop(label_name, axis=1), True)
        # print(train_frame)
        # no need cv - overfit also okay - memorize it also okay
        self.learner.fit(train_frame, lat_frame)

    def predict(self, skus: List[str], collectives: List[str], world_size: List[int], buffer_size: List[int], param_dtype: List[str]) -> List[float]:
        data_as_dict = {
            "sku": skus,
            "collectives": collectives,
            "world_size": world_size,
            "buffer_size": buffer_size,
            "param_dtype": param_dtype
        }

        df = self.preproc(pandas.DataFrame(data_as_dict))
        # print(data_as_dict)
        return self.learner.predict(df).tolist()

    def predict_single(self, sku: str, collectives: str, world_size: int, buffer_size: int, param_dtype: str) -> float:
        return self.predict([sku], [collectives], [world_size], [buffer_size], [param_dtype])[0]

# local test codes


class ExecutionEnvironment:
    def __init__(self, local_gpu_count: int, total_host_count: int, sku: str) -> None:
        self.local_gpu_count = local_gpu_count
        self.total_host_count = total_host_count
        self.sku = sku

    @property
    def world_size(self) -> float:
        return float(self.local_gpu_count * self.total_host_count)

    def get_allgather_reducescatter_world_size(self, sharding_strategy: ShardingStrategy) -> int:
        if sharding_strategy == ShardingStrategy.NO_SHARD:
            return 1
        elif sharding_strategy == ShardingStrategy.FULL_SHARD or sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            return float(self.world_size)
        else:
            return float(self.local_gpu_count)

    def get_allreduce_world_size(self, sharding_strategy: ShardingStrategy) -> float:
        if sharding_strategy == ShardingStrategy.NO_SHARD:
            return 0
        elif sharding_strategy == ShardingStrategy.FULL_SHARD or sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            return 0
        else:
            return float(self.total_host_count)


@dataclass
class ModuleExecutionDescriptor:
    name: str = ""
    activation_count: float = 0  # gb
    activation_dtype: str = "float"
    param_count: int = 0
    latency: float = 0
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    full_optimizer_latency: float = 0
    optimizer_memory: float = 0  # gb
    param_dtype: str = "float"

    def get_idle_memory_usage(self, exec_env: ExecutionEnvironment) -> float:
        return dtype2size[self.param_dtype] * self.param_count / exec_env.get_allgather_reducescatter_world_size(self.sharding_strategy)

    def get_activation_memory_usage(self, exec_env: ExecutionEnvironment) -> float:
        return dtype2size[self.param_dtype] * self.activation_count

    def get_active_to_inactive_weight_memory_delta_fwd(self, exec_env: ExecutionEnvironment) -> float:
        if self.sharding_strategy == ShardingStrategy.NO_SHARD or self.sharding_strategy == ShardingStrategy._HYBRID_SHARD_ZERO2 or self.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            return 0  # Zero2 retains memory weights. No memory transition
        else:
            # free weights memory
            world_size = exec_env.get_allgather_reducescatter_world_size(
                self.sharding_strategy)
            return dtype2size[self.param_dtype] * self.param_count * (world_size - 1) / world_size

    def get_inactive_to_active_weight_memory_delta_fwd(self, exec_env: ExecutionEnvironment) -> float:
        # requires allgather, unless you're noshard
        if self.sharding_strategy == ShardingStrategy.NO_SHARD:
            return 0
        else:
            # gathers remote memory
            world_size = exec_env.get_allgather_reducescatter_world_size(
                self.sharding_strategy)
            return dtype2size[self.param_dtype] * self.param_count * (world_size - 1) / world_size

    def get_active_to_inactive_weight_memory_delta_bwd(self, exec_env: ExecutionEnvironment) -> float:
        # only noshard frees no memory
        if self.sharding_strategy == ShardingStrategy.NO_SHARD:
            return 0
        else:
            world_size = exec_env.get_allgather_reducescatter_world_size(
                self.sharding_strategy)
            return dtype2size[self.param_dtype] * self.param_count * (1 - 1 / world_size)

    def get_inactive_to_active_weight_memory_delta_bwd(self, exec_env: ExecutionEnvironment) -> float:
        if self.sharding_strategy == ShardingStrategy.NO_SHARD or self.sharding_strategy == ShardingStrategy._HYBRID_SHARD_ZERO2 or self.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            return 0  # ZERO2 retains memory weights
        else:
            return dtype2size[self.param_dtype] * self.param_count * (1 - 1 / exec_env.get_allgather_reducescatter_world_size(self.sharding_strategy))

    dbg__layer_variable: ClassVar[int] = 0

    @staticmethod
    def dbg__get_random_exec_desc():
        ModuleExecutionDescriptor.dbg__layer_variable += 1
        import random
        return ModuleExecutionDescriptor(
            f"layer_{ModuleExecutionDescriptor.dbg__layer_variable}", random.randint(10, 100), "float", random.randint(10, 100), random.randint(10, 100), random.choice([x for x in ShardingStrategy]), random.randint(0, 100), random.randint(0, 100), "float")


class OSDPExecutionSimulator:
    def __init__(self, comms_perf_model: CommsPerformanceModel, compute_perf_model: List[ModuleExecutionDescriptor]):
        self.comms_perf_model = comms_perf_model
        self.compute_perf_model = compute_perf_model

    # latency (s), memory (gib)
    def simulate(self, exec_env: ExecutionEnvironment) -> Tuple[float, float]:
        # assume single comms stream
        # assume single comp stream
        # assume very fast CPU
        # assume prefetcher is on

        comp_ts = 0 # time ticks at which the compute stream is ready to take new task
        comm_ts = 0 # time ticks at which the communication stream is ready to take new task

        activation_memory = 0
        weights_memory = sum(desc.get_idle_memory_usage(exec_env)
                             for desc in self.compute_perf_model)  # base memory usage
        optimizer_memory = 2 * weights_memory  # ADAM-style
        gradient_memory = 0

        # ignores collective memory - assume the collectives directly takes from the original tensor.

        max_memory_usage = activation_memory + \
            weights_memory + optimizer_memory + gradient_memory
        
        # maximum memory usage is the sum of idle memory at this point
        # forward pass:
        for idx, exec_desc in enumerate(self.compute_perf_model):
            if exec_desc.sharding_strategy != ShardingStrategy.NO_SHARD:
                world_size = exec_env.get_allgather_reducescatter_world_size(
                    exec_desc.sharding_strategy)

                comm_lat = self.comms_perf_model.predict_single(
                    exec_env.sku, "allgather", world_size, exec_desc.param_count, exec_desc.param_dtype
                )
                comm_ts += comm_lat
                # allgather for layer idx will finish at comm_ts

            # compute stream will start compute layer idx 
            # when comm for layer idx is finished and when the compute stream is idle
            comp_ts = max(comp_ts, comm_ts) + exec_desc.latency

            # compute memory watermarks
            activation_memory += exec_desc.get_activation_memory_usage(
                exec_env)
            # full parameter size
            weights_memory += exec_desc.get_inactive_to_active_weight_memory_delta_fwd(
                exec_env)
            max_memory_usage = max(max_memory_usage, activation_memory +
                                   weights_memory + optimizer_memory + gradient_memory)

            weights_memory -= exec_desc.get_active_to_inactive_weight_memory_delta_fwd(
                exec_env)

        # backward
        # set communication tick to the end of compute
        # note compute always finishes after comms in the forward pass.
        comm_ts = comp_ts

        for idx in range(len(self.compute_perf_model) - 1, -1, -1):
            exec_desc = self.compute_perf_model[idx]
            world_size = exec_env.get_allgather_reducescatter_world_size(
                exec_desc.sharding_strategy)

            if exec_desc.sharding_strategy == ShardingStrategy.FULL_SHARD or exec_desc.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
                # I need allgather?
                allgather_lat = self.comms_perf_model.predict_single(
                    exec_env.sku, "allgather", world_size, exec_desc.param_count, exec_desc.param_dtype)
                comm_ts += allgather_lat

            # execute
            weights_memory += exec_desc.get_inactive_to_active_weight_memory_delta_bwd(
                exec_env)
            gradient_memory += exec_desc.get_idle_memory_usage(exec_env)
            max_memory_usage = max(max_memory_usage, activation_memory +
                                   weights_memory + optimizer_memory + gradient_memory)
            
            comp_ts = max(comp_ts, comm_ts) + exec_desc.latency
            # release weight memory if needed
            weights_memory -= exec_desc.get_inactive_to_active_weight_memory_delta_bwd(
                exec_env)

            # issue reducescatter if needed.
            if exec_desc.sharding_strategy != ShardingStrategy.NO_SHARD:
                rs_lat = self.comms_perf_model.predict_single(
                    exec_env.sku, "reduce_scatter", world_size, exec_desc.param_count, exec_desc.param_dtype)
                # reducescatter for layer idx cannot happen before this layer's compute.
                comm_ts = max(comm_ts, comp_ts) + rs_lat

            # issue allreduce if needed
            if exec_desc.sharding_strategy == ShardingStrategy.HYBRID_SHARD or exec_desc.sharding_strategy == ShardingStrategy._HYBRID_SHARD_ZERO2:
                ar_lat = self.comms_perf_model.predict_single(
                    exec_env.sku, "allreduce", world_size, exec_desc.param_count, exec_desc.param_dtype)
                # allreduce can happen right after reducescatter
                comm_ts += ar_lat

        # compute optimizer latency
        # assuming latency is porportional to # of parameters locally.
        opt_lat = 0
        for exec_desc in self.compute_perf_model:
            world_size = exec_env.get_allgather_reducescatter_world_size(
                exec_desc.sharding_strategy)
            opt_lat = exec_desc.full_optimizer_latency / world_size

        # optimizer and backward has a bogus dependency.
        return max(comm_ts, comp_ts) + opt_lat, max_memory_usage


### LOCAL TESTS ###
ds_d = {
    "sku": ["SMC", "ZIONEX", "ZIONEX_80G", "SMC_A100_80GB"] * 2,
    "collectives": ["reduce_scatter", "allgather", "allreduce", "allgather"] * 2,
    "world_size": [float(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]],
    "buffer_size": [float(x) for x in [4096, 8192, 2048, 1024, 4, 8, 16, 32]],
    "param_dtype": ["float"] * 8,
    "latency": [1, 2, 0.5, 0.25, 0.001, 0.002, 0.004, 0.008],
}

df = pandas.DataFrame(ds_d)
# print(df)

perf_model = CommsPerformanceModel(df)
dummy_prediction_input = {
    "skus": ["SMC", "ZIONEX_80G"],
    "collectives": ["reduce_scatter", "allgather"],
    "world_size": [8.0, 8.0],
    "buffer_size": [4096.0, 8192.0],
    "param_dtype": ["float"] * 2,
}

estimation = perf_model.predict(**dummy_prediction_input)
# print(estimation)

dummy_compute_perf_model = [
    ModuleExecutionDescriptor.dbg__get_random_exec_desc() for _ in range(2)
]


sim = OSDPExecutionSimulator(perf_model, dummy_compute_perf_model)
exec_env = ExecutionEnvironment(8, 16, "SMC")
lat, mem = sim.simulate(exec_env)

print(lat, mem)
