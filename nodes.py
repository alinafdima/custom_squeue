# ------------------------------------------------------------------------------
# File: nodes.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Mon Aug 12 2024
#
# Parses sinfo and returns a list of nodes with their GPU type and count.
#
# ------------------------------------------------------------------------------
# %%
from dataclasses import dataclass
import re
import subprocess

down_states = ["down", "drain", "fail", "fail*", "down*", "draining", "drained"]


@dataclass
class Node:
    name: str
    gpu_type: str
    gpu_count: int
    state: str
    is_unavailable: bool
    partition: str
    is_asteroid: bool
    is_private: bool

    def __repr__(self):
        return (
            f"Name: {self.name}".ljust(25)
            + f"GPUs: {self.gpu_count}({self.gpu_type})".ljust(30)
            + f"State: {self.state}".ljust(20)
            + ("asteroid" if self.is_asteroid else "universe").ljust(10)
            + ("private" if self.is_private else "public").ljust(10)
        )


def parse_complex_gpu_descriptions(gpu_raw, node_name):
    """
    Parses gpu info. Can handle nodes that contain multiple GPU types.
    gpu_raw is obtained from sinfo
    Example string from a node with 2 gpu types:
    'gpu:RTX8000:2,gpu:RTX6000:2'

    Args:
        gpu_raw (str): The raw string containing the GPU info
        node_name (str): The name of the node

    Raises:
        NotImplementedError: In case the info does not match the expected format.

    Returns:
        Tuple(int, str): Total number of nodes and their type
    """
    gres_regex = r"gpu:(\w*):(\d)(\(?.*\)?)"

    individual_gpus = []
    for current_gpu_raw in gpu_raw.split(","):
        matches = re.match(gres_regex, current_gpu_raw)
        if not matches:
            raise NotImplementedError(
                f"Could not parse GPU info for node {node_name}: "
                + f"{current_gpu_raw}"
                "Please check the sinfo output."
            )
        else:
            gpu_type, gpu_count, _ = matches.groups()
            individual_gpus.append((gpu_type, int(gpu_count)))

    gpu_count = sum([count for _, count in individual_gpus])
    gpu_type = ",".join([f"{y}x{x}" for x, y in individual_gpus])
    return gpu_count, gpu_type


class NodeMaster:
    def __init__(self) -> None:
        sinfo_output = (
            subprocess.check_output(["sinfo", "-N", "-o", "%N||%T||%P||%G"])
            .decode("utf-8")
            .strip()
            .split("\n")[1:]
        )
        nodes = []
        for node_raw in sinfo_output:
            try:
                node_name, status, partition, gpu_raw = [
                    x for x in node_raw.split("||") if x
                ]
            except ValueError:
                print(f"Could not parse node info: {node_raw}")
                continue

            gpu_count, gpu_type = parse_complex_gpu_descriptions(
                gpu_raw, node_name
            )
            is_private = partition not in [
                "asteroids",
                "asteroids*",
                "universe",
                "universe*",
            ]
            is_unavailable = status in down_states or is_private
            nodes.append(
                Node(
                    node_name,
                    gpu_type=gpu_type,
                    gpu_count=int(gpu_count),
                    state=status,
                    is_unavailable=is_unavailable,
                    partition=partition,
                    is_asteroid=partition == "asteroids",
                    is_private=is_private,
                )
            )

        self.nodes = sorted(
            nodes,
            key=lambda x: (
                # Cluster nodes first, asteroids last
                x.is_unavailable,  # Available nodes first
                -x.gpu_count,  # Sort by gpu count in descending order
                x.gpu_type,  # Sort by GPU type in ascending order
                x.name,  # Sort by node name in ascending order
            ),
        )
        self.total_gpu_count = sum([node.gpu_count for node in self.nodes])
        self.total_gpu_count_available = sum(
            [node.gpu_count for node in self.nodes if not node.is_unavailable]
        )
        self.total_non_asteroid_count = sum(
            [
                node.gpu_count
                for node in self.nodes
                if not node.is_asteroid and not node.is_unavailable
            ]
        )

    def __repr__(self):
        ret = f"Total GPUs: {self.total_gpu_count}\n"
        ret += (
            f"Total GPUs currently online: {self.total_gpu_count_available}\n"
        )
        ret += (
            f"Total non-asteroid GPUs: {self.total_non_asteroid_count}\n"
        )
        ret += "\nPublic Nodes:\n"
        for node in self.nodes:
            if node.is_unavailable:
                continue
            ret += str(node) + "\n"
        return ret


if __name__ == "__main__":
    nodes = NodeMaster()
    public_node_names = [
        node.name for node in NodeMaster().nodes if not node.is_private
    ]
    # print("Private nodes:")
    # for node in [node for node in nodes.nodes if node.is_unavailable]:
    #     print(node)

    # print("\nPublic nodes:")
    # for node in [node for node in nodes.nodes if not node.is_unavailable]:
    #     print(node)
    print(nodes)


