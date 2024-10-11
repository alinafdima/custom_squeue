# Custom squeue
Custom squeue for a slurm server

Parses the output of `scontrol show job -d` and `sinfo` and shows the user-relevant information, featuring a GPU count of the cluster resources and per-user information.

When run with the `--default` settings it will show a breakdown of free and utilized resources (nodes), the pending, running and past jobs of the current user, an overview of the number of resources used by each other user and the incoming load on the cluster. 

When run with the flag `--jobs` it will show all of the running + pending jobs on the cluster, similar to `squeue`.

It can also be used interactively as a python library, using the output of `JobMaster` to inspect all of the cluster jobs:
```
from job_master import JobMaster
jobs = JobMaster()
```

The nodes can be interacted with in a similar way:
```
from nodes import NodeMaster
nodes = NodeMaster()
print(nodes)
```

## Example usage:
```` bash
python custom_squeue_v2.py --default
````


## Example entry in .bashrc
```` bash
alias custom_squeue="watch -n 2 python <PATH_TO_FOLDER>/custom_squeue_v2.py --default"
````