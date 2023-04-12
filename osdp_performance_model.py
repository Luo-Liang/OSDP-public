from collections import defaultdict, deque
from dataclasses import dataclass
import sys
from typing import ClassVar, List, Optional, Tuple
import pandas
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.distributed.fsdp.api import ShardingStrategy
from sortedcontainers import SortedList

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
            return 1
        elif sharding_strategy == ShardingStrategy.FULL_SHARD or sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            return 1
        else:
            return float(self.total_host_count)


class list_of_tuple_with_pos_idx(list):
    def __init__(self, max_idx, *args, **kwargs) -> None:
        self.max_idx = max_idx
        super().__init__(*args, **kwargs)

    def append(self, item):
        assert type(item) is tuple
        _, idx = item
        if idx < 0 or idx > self.max_idx:
            # invalid
            return
        else:
            super().append(item)


def n_minus_1_over_n(n):
    return (n - 1) / n


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

    def needs_allgather_forward(self):
        return self.sharding_strategy != ShardingStrategy.NO_SHARD

    def needs_allgather_backward(self):
        return self.sharding_strategy == ShardingStrategy.FULL_SHARD or self.sharding_strategy == ShardingStrategy.HYBRID_SHARD

    def needs_reduce_scatter_backward(self):
        return self.sharding_strategy != ShardingStrategy.NO_SHARD

    def needs_allreduce_backward(self):
        return self.sharding_strategy == ShardingStrategy.HYBRID_SHARD or self.sharding_strategy == ShardingStrategy._HYBRID_SHARD_ZERO2 or self.sharding_strategy == ShardingStrategy.NO_SHARD

    def _weights_memory_delta(self):
        if self.needs_allgather_forward():
            world_size = exec_env.get_allgather_reducescatter_world_size(
                self.sharding_strategy)
            return self.param_count * dtype2size[self.param_dtype] * n_minus_1_over_n(world_size)
        else:
            return 0

    def _activation_memory_delta(self):
        return self.activation_count * dtype2size[self.param_dtype]

    def _gradient_memory_delta(self):
        return self._activation_memory_delta()

    dbg__layer_variable: ClassVar[int] = 0

    @staticmethod
    def dbg__get_random_exec_desc():
        ModuleExecutionDescriptor.dbg__layer_variable += 1
        import random
        return ModuleExecutionDescriptor(
            f"layer_{ModuleExecutionDescriptor.dbg__layer_variable}", random.randint(10, 100), "float", random.randint(10, 100), random.randint(10, 100), ShardingStrategy.NO_SHARD, random.randint(0, 100), random.randint(0, 100), "float")


class OSDPExecutionSimulator:
    def __init__(self, comms_perf_model: CommsPerformanceModel, compute_perf_model: List[ModuleExecutionDescriptor]):
        self.comms_perf_model = comms_perf_model
        self.compute_perf_model = compute_perf_model
        self.dummy_event = None

    def get_latency(self, exec_env: ExecutionEnvironment, bundle: Tuple[str, int]):
        if bundle == self.dummy_event:
            return sys.maxsize

        task_type, exec_desc_idx = bundle
        exec_desc = self.compute_perf_model[exec_desc_idx]
        if "compute" in task_type:
            return exec_desc.latency
        elif "optimizer" in task_type:
            world_size = exec_env.get_allgather_reducescatter_world_size(
                exec_desc.sharding_strategy)
            return exec_desc.full_optimizer_latency / world_size
        elif "allreduce" in task_type:
            return self.comms_perf_model.predict_single(exec_env.sku, "allreduce", exec_env.get_allgather_reducescatter_world_size(exec_desc.sharding_strategy), exec_desc.param_count, exec_desc.param_dtype)
        else:
            collectives_name = task_type.split(":")[0]
            return self.comms_perf_model.predict_single(exec_env.sku, collectives_name, exec_env.get_allgather_reducescatter_world_size(exec_desc.sharding_strategy), exec_desc.param_count, exec_desc.param_dtype)

    # Given an execution environment,
    # use the learned communication performance model
    # and the profiled compute performance model to derive a performance estimation
    # returns latency (s), memory (gib)
    def simulate(self, exec_env: ExecutionEnvironment, backward_prefetch: int = 1, limit_allgather: int = 1000) -> Tuple[float, float]:
        # assume single comms stream
        # assume single comp stream
        # assume very fast CPU
        # assume prefetcher is on

        TASKTYPE_ALLGATHER_FWD = "allgather:fwd"
        TASKTYPE_COMPUTE_FWD = "compute:fwd"
        TASKTYPE_ALLGATHER_BWD = "allgather:bwd"
        TASKTYPE_COMPUTE_BWD = "compute:bwd"
        TASKTYPE_REDUCESCATTER_BWD = "reduce_scatter:Bwd"
        TASKTYPE_ALLREDUCE_BWD = "allreduce:bwd"
        TASKTYPE_OPTIMIZER = "optimizer"

        TASKSTREAM_COMPUTE = "compute_stream"
        TASKSTREAM_COMMS = "comms_stream"

        # first, gather all tasks
        deps = defaultdict(lambda: list_of_tuple_with_pos_idx(
            max_idx=len(self.compute_perf_model) - 1))
        # on the forward pass, for each compute,
        # create at most one allgather task

        # first, build dependency graph and use the actual
        # event serialization in the current FSDP's impl

        compute_stream = list_of_tuple_with_pos_idx(
            len(self.compute_perf_model) - 1)
        comms_stream = list_of_tuple_with_pos_idx(
            len(self.compute_perf_model) - 1)

        for idx, exec_desc in enumerate(self.compute_perf_model):
            comp_task = (TASKTYPE_COMPUTE_FWD, idx)
            allgather_task = (TASKTYPE_ALLGATHER_FWD, idx)
            reduce_scatter_task = (TASKTYPE_REDUCESCATTER_BWD, idx)
            comp_bwd_task = (TASKTYPE_COMPUTE_BWD, idx)
            allgather_bwd_task = (TASKTYPE_ALLGATHER_BWD, idx)
            allreduce_task = (TASKTYPE_ALLREDUCE_BWD, idx)

            prev_comp_task = (TASKTYPE_COMPUTE_FWD, idx - 1)
            # prev_allgather_task = (TASKTYPE_ALLGATHER_FWD, idx - 1)
            # next_allgather_task_bwd = (TASKTYPE_ALLGATHER_BWD, idx + 1)
            next_comp_task_bwd = (TASKTYPE_COMPUTE_BWD, idx + 1)
            last_layer_compute = (TASKTYPE_COMPUTE_FWD,
                                  len(self.compute_perf_model) - 1)

            optimizer_task = (TASKTYPE_OPTIMIZER, idx)
            # forward
            if exec_desc.needs_allgather_forward():
                # needs to finish allgather before compute can happen
                deps[comp_task].append(allgather_task)
                # provides serialization for comms stream
                comms_stream.append(allgather_task)

            # depends on previous compute
            # true dependency
            deps[comp_task].append(prev_comp_task)

            # backward
            # current compute depends on next layer's compute
            deps[comp_bwd_task].append(next_comp_task_bwd)
            deps[comp_bwd_task].append(last_layer_compute)

            if exec_desc.needs_allgather_backward():
                # allgather backwad cannot happen before fwd of last layer
                deps[allgather_bwd_task].append(last_layer_compute)
                deps[comp_bwd_task].append(allgather_bwd_task)

            if exec_desc.needs_reduce_scatter_backward():
                deps[reduce_scatter_task].append(comp_bwd_task)
                deps[optimizer_task].append(reduce_scatter_task)

            if exec_desc.needs_allreduce_backward():
                if exec_desc.sharding_strategy == ShardingStrategy.NO_SHARD:
                    # ddp
                    deps[allreduce_task].append(comp_bwd_task)
                else:
                    # hybrid shard
                    deps[allreduce_task].append(reduce_scatter_task)
                
                deps[optimizer_task].append(allreduce_task)

            # hook up dependency for optimizer

            # provide serialization for the forward pass compute
            compute_stream.append(comp_task)

        # provide serialization for backward
        for idx in range(len(self.compute_perf_model) - 1, -1, -1):
            # backward
            exec_desc = self.compute_perf_model[idx]
            reduce_scatter_task = (TASKTYPE_REDUCESCATTER_BWD, idx)
            comp_bwd_task = (TASKTYPE_COMPUTE_BWD, idx)
            allgather_bwd_task = (TASKTYPE_ALLGATHER_BWD, idx)
            allreduce_task = (TASKTYPE_ALLREDUCE_BWD, idx)
            # special handle prefetch

            # mimic backward prefetch
            if exec_desc.needs_allgather_backward():
                comms_stream.append(allgather_bwd_task)
                prefetch_idx = idx - 1
                while backward_prefetch > 0 and idx == len(self.compute_perf_model) - 1 and prefetch_idx >= 0 and self.compute_perf_model[prefetch_idx].needs_allgather_backward():
                    # issue one more allgather
                    next_allgather_task = (TASKTYPE_ALLGATHER_BWD, prefetch_idx)
                    comms_stream.append(next_allgather_task)
                    backward_prefetch -= 1
                    prefetch_idx -= 1


            # issue reducescatter for this layer if needed
            if exec_desc.needs_reduce_scatter_backward():
                comms_stream.append(reduce_scatter_task)

            # issue allreduce for this layer if needed
            if exec_desc.needs_allreduce_backward():
                comms_stream.append(allreduce_task)

            compute_stream.append(comp_bwd_task)

        # then, queue optimizer tasks
        for idx in range(len(self.compute_perf_model) - 1, -1, -1):
            optimizer_task = (TASKTYPE_OPTIMIZER, idx)
            compute_stream.append(optimizer_task)

        # now compute and comms stream represent a
        # real world serialization for the dependency we have.
        # the final step is to really simulate them

        # note that as long as the comms stream is nonempty, the process isn't done and won't deadlock
        finish_time = {}
        pending_allgathers_fwd = 0
        # active_tasks = defaultdict(lambda: 0)
        # a task is scheduable iff
        # (1) the task has all explicit dependencies resolved
        # (2) the task has all implicit dependencies resolved (single stream semantics)
        # (3) additional constraints such as number of inflight allgathers

        def schedulable(task):
            if task == self.dummy_event:
                return False

            ready = True
            for dep in deps[task]:
                if dep not in finish_time:
                    ready = False
                    break

            task_type = task[0]
            if "allgather" in task_type and pending_allgathers_fwd >= limit_allgather:
                # cannot schedule more than allowed pending allgathers
                return False

            return ready

        # this is a general implementation, for single stream execution this is not needed
        EVENT_SCHEDULED = 2
        EVENT_START = 1
        EVENT_END = 0  # prioritize end events

        # event-based memory profiling hooks
        ACTIVATION_MEMORY = "activation_memory"
        WEIGHTS_MEMORY = "weights_memory"
        OPTIMIZER_MEMORY = "optimizer_memory"
        GRADIENT_MEMORY = "gradient_memory"

        memory_profile = {
            ACTIVATION_MEMORY: 0,
            WEIGHTS_MEMORY: sum(desc.get_idle_memory_usage(exec_env) for desc in self.compute_perf_model),
            OPTIMIZER_MEMORY: 2 * sum(desc.get_idle_memory_usage(exec_env) for desc in self.compute_perf_model),
            GRADIENT_MEMORY: 0,
        }

        # provide transition actions for all of the following:
        # TASKTYPE_ALLGATHER_FWD = "allgather:fwd"
        # TASKTYPE_COMPUTE_FWD = "compute:fwd"
        # TASKTYPE_ALLGATHER_BWD = "allgather:bwd"
        # TASKTYPE_COMPUTE_BWD = "compute:bwd"
        # TASKTYPE_REDUCESCATTER_BWD = "reduce_scatter:Bwd"
        # TASKTYPE_ALLREDUCE_BWD = "allreduce:bwd"
        # TASKTYPE_OPTIMIZER = "optimizer"

        max_memory_usage = sum(memory_profile.values())

        def event_no_op(exec_desc):
            pass

        def _record_memory_footprint():
            nonlocal max_memory_usage
            max_memory_usage = max(
                max_memory_usage, sum(memory_profile.values()))

        def event_start_allgather(exec_desc):
            memory_profile[WEIGHTS_MEMORY] += exec_desc._weights_memory_delta()
            _record_memory_footprint()

        def event_start_compute(exec_desc):
            memory_profile[ACTIVATION_MEMORY] += exec_desc._activation_memory_delta()
            _record_memory_footprint()

        def event_end_reduce_scatter(exec_desc):
            memory_profile[WEIGHTS_MEMORY] -= exec_desc._weights_memory_delta()
            _record_memory_footprint()

        def event_start_compute_bwd(exec_desc):
            memory_profile[ACTIVATION_MEMORY] += exec_desc._gradient_memory_delta
            _record_memory_footprint()

        memory_profiler_hooks = defaultdict(lambda: event_no_op)

        # when does weight memory increase?
        # when allgather is scheduled (acquire) fwd, bwd,
        memory_profiler_hooks[(
            EVENT_SCHEDULED, TASKTYPE_ALLGATHER_FWD)] = event_start_allgather
        memory_profiler_hooks[(
            EVENT_SCHEDULED, TASKTYPE_ALLGATHER_BWD)] = event_start_allgather

        # when does weight memory decrease?
        # when reducescatter has finished running
        memory_profiler_hooks[(
            EVENT_END, TASKTYPE_REDUCESCATTER_BWD)] = event_end_reduce_scatter

        # when does activation memory increase?
        # when compute fwd is active
        memory_profiler_hooks[(
            EVENT_START, TASKTYPE_COMPUTE_FWD)] = event_start_compute

        # when does gradient memory increase?
        # when backward compute is active
        memory_profiler_hooks[(
            EVENT_START, TASKTYPE_COMPUTE_BWD)] = event_start_compute_bwd

        def get_task_stream(task):
            task_name = task[0]
            if "compute" in task_name or "optimizer" in task_name:
                return TASKSTREAM_COMPUTE
            else:
                return TASKSTREAM_COMMS

        def get_earliest_possible_schedule_time(task, proposal):
            print(
                f"[???] probing for schedule time {task}. proposal = {proposal}")
            for dep in deps[task]:
                proposal = max(proposal, finish_time[dep])
                print(f"    dep: {dep}, finished = {finish_time[dep]}")

            return proposal

        timestamps = defaultdict(lambda: 0)

        print(f"all dependencies {deps}")
        print(
            f"strategies = {[desc.sharding_strategy for desc in self.compute_perf_model]}")
        print(f"comms stream = {comms_stream}")
        print(f"comps stream = {compute_stream}")

        timeline = SortedList()

        while comms_stream or compute_stream or timeline:
            # any task already queued?
            if timeline:
                # timeline is an arbitrary serialization of
                # execution of compute and comms stream.
                schedule_ts, event_status, curr_task = timeline.pop(0)
                done_task_type, _ = curr_task
                if "allgather" in done_task_type:
                    if event_status == EVENT_START:
                        pending_allgathers_fwd += 1
                    else:
                        pending_allgathers_fwd -= 1

                # is this a start event?
                # if so, queue an event end
                if event_status == EVENT_START:
                    # catch up with global clock
                    start_time = max(timestamps[get_task_stream(
                        curr_task)], schedule_ts)

                    end_time = start_time + self.get_latency(exec_env, curr_task)

                    timeline.add(
                        (start_time + self.get_latency(exec_env, curr_task), EVENT_END, curr_task))
                    memory_profiler_hooks[(EVENT_START, curr_task)](curr_task)
                    # advance time stamp, because i know this stream
                    # cannot do anything else between now and now + latency
                    # advance my time so my task is not "preempted"
                    print(f"[{start_time}] {curr_task} started. Will finish at {end_time}")

                    timestamps[get_task_stream(
                        curr_task)] = end_time

                else:
                    # mark this task as done
                    finish_time[curr_task] = timestamps[get_task_stream(
                        curr_task)]
                    memory_profiler_hooks[(EVENT_END, curr_task)](curr_task)

                    print(
                        f"[{timestamps[get_task_stream(curr_task)]}] {curr_task} finished.")

            while compute_stream and schedulable(compute_stream[0]):
                comp_task = compute_stream.pop(0)
                memory_profiler_hooks[(EVENT_SCHEDULED, comp_task)](comp_task)
                schedule_time = get_earliest_possible_schedule_time(
                    comp_task, timestamps[TASKSTREAM_COMPUTE])

                print(
                    f"[{schedule_time}] scheduled comp task {comp_task}")

                timeline.add(
                    (schedule_time, EVENT_START, comp_task))

            while comms_stream and schedulable(comms_stream[0]):
                # issue as many comms tasks as possible
                comms_task = comms_stream.pop(0)
                memory_profiler_hooks[(EVENT_SCHEDULED, comms_task)](
                    comms_task)

                schedule_time = get_earliest_possible_schedule_time(
                    comms_task, timestamps[TASKSTREAM_COMMS])

                print(
                    f"[{schedule_time}] scheduled comms task {comms_task}")

                timeline.add(
                    (schedule_time, EVENT_START, comms_task)
                )

        # optimizer and backward has a bogus dependency.
        return max(timestamps.values()), max_memory_usage


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
