import torch
import time
device = torch.device("mps")


def probe_lat_batched(batch, dim, cnt, reps):
    tensors = [torch.empty((batch, dim)).to(device) for _ in range(cnt)]
    start = time.perf_counter()
    for _ in range(reps):
        for i in range(cnt):
            concat = torch.cat(tensors[: i + 1])
    end = time.perf_counter()
    return (end - start) / reps


def probe_lat_sequential(batch, dim, cnt, reps):
    tensors = [torch.empty((batch, dim)).to(device) for _ in range(cnt)]
    start = time.perf_counter()
    for _ in range(reps):
        ret = tensors[0]
        for i in range(1, cnt):
            ret = torch.cat([ret, tensors[i]])
    end = time.perf_counter()
    return (end - start) / reps


batch = 1024
reps = 10
for dim in range(1, 16, 4):
    print(f"DIM = {dim}")
    for cnt in range(1, 51):
        batched_lat = probe_lat_batched(batch, dim, cnt, reps)
        sequential_lat = probe_lat_sequential(batch, dim, cnt, reps)
        print(
            f"  cnt = {cnt}, batched = {batched_lat}, seq = {sequential_lat}")

# conclusion: there is never a case on GPU that sequential is faster than batched
# there're some cases where batched is slower than sequential on CPU however.