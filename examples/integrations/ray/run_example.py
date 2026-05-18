"""Runnable Ray example with the Ingero task annotation helper.

Launches a handful of Ray tasks, each annotated into a live Ingero
trace, so every task's eBPF events can be sliced per task afterwards.

Run the agent first, in another shell, so the annotation socket exists:

    sudo ingero trace --record --annotate

Then run this script:

    python run_example.py

Afterwards, slice the recorded trace by task:

    ingero query --label task_id=matmul-3
    ingero explain --label task_name=matmul

If the agent is not running with `--annotate`, the helper degrades to a
no-op and the Ray job is unaffected.

Requires: ray (see requirements.txt).
"""

from __future__ import annotations

import ray

from ingero_ray import annotate_task, task_annotation


@ray.remote
@annotate_task(task_name="matmul")
def matmul(n: int):
    """A decorator-annotated task: annotated automatically on each run."""
    import random

    size = 64
    a = [[random.random() for _ in range(size)] for _ in range(size)]
    total = 0.0
    for _ in range(n):
        for row in a:
            total += sum(row)
    return total


@ray.remote
def reduce_shard(shard_id: int):
    """A context-manager-annotated task: annotated inside the body."""
    with task_annotation(task_name="reduce", labels={"shard": str(shard_id)}):
        return sum(range(shard_id * 1000, (shard_id + 1) * 1000))


def main() -> None:
    ray.init()
    try:
        matmul_results = ray.get([matmul.remote(50) for _ in range(4)])
        reduce_results = ray.get([reduce_shard.remote(i) for i in range(4)])
        print("matmul results:", matmul_results)
        print("reduce results:", reduce_results)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
