# OSDP-public
Composable + Tunable = Optimal

OSDP = 
1. A learned communication performance (GBDT) model based on real world measurements
2. A profiled compute performance model based on memory and latency profile for the module
3. A Simulator that computes latency and peak memory usage given these two performance models'
The sim is needed because launching distributed jobs are too slow - this surrogate model allows the tuner to explore a larger space
4. A sequential model optimizer that explores the exploration space.

The core of the OSDP performance model is similar to the one used in Srifty:

@article{luo2022srifty,
  title={SRIFTY: Swift and Thrifty Distributed Neural Network Training on the Cloud},
  author={Luo, Liang and West, Peter and Patel, Pratyush and Krishnamurthy, Arvind and Ceze, Luis},
  journal={Proceedings of Machine Learning and Systems},
  volume={4},
  pages={833--847},
  year={2022}
}

Output = 
A list of ShardingStrategy which corresponds to the optimal FSDP shardingstrategy given a list of modules (abstracted as a serialized list of execution information).
