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
            + f"GPUs: {self.gpu_count}({self.gpu_type})".ljust(50)
            + f"State: {self.state}".ljust(20)
            + ("asteroid" if self.is_asteroid else "universe").ljust(10)
            + ("private" if self.is_private else "public").ljust(10)
        )


def parse_complex_gpu_descriptions(gpu_raw, node_name):
    """
    Parses gpu info. Can handle nodes that contain multiple GPU types.
    gpu_raw is obtained from sinfo, and it contains a list of 
    GPU specs (gpu:) followed by a list of corresponding memory specs (gpumem:).
    Example string from a node with 2 gpu types (name1 and name2) with 2 GPUs each:
    'gpu:name1:2(S:0-15),gpu:name2:2(S:0-15),gpumem:name1:48G,gpumem:name2:48G'
    Name convention changed to: 
    'gpu:name1:2(S:0-15),gpu:name2:2(S:0-15),gpumem:name1:no_consume:48G,gpumem:name2:no_consume:48G'

    Args:
        gpu_raw (str): The raw string containing the GPU info
        node_name (str): The name of the node

    Raises:
        ValueError: In case the info does not match the expected format.

    Returns:
        Tuple(int, str): Total number of nodes and their type
    """
    # Step 1: Parse the gpu_raw string into a list of entities corresponding 
    # to each gpu type its respective memory spec (2 entities per gpu type)

    # The entities are separated by commas, but the socket info can also contain commas
    # 1a. Split by gpu: and gpumem:
    entities = re.split(r"gpu:|gpumem:", gpu_raw)[1:]
    # 1b. Remove final comma if present
    entities = [e[:-1] if e[-1] == "," else e for e in entities]
    # 1c. Check if the number of entities is even
    if len(entities) % 2 == 1:
        raise ValueError(
            f"Node GPU parsing error: Odd number of entities: {len(entities)}"
        )
    T = len(entities) // 2
    # 1d. Double check the format by adding "gpu:" and "gpumem:" back to the entities
    recon_gpu_raw = ",".join(
        ["gpu:" + e for e in entities[:T]]
        + ["gpumem:" + e for e in entities[T:]]
    )
    if recon_gpu_raw != gpu_raw:
        raise ValueError(
            "Node GPU parsing error: "
            + f"Expected format: {recon_gpu_raw}"
            + f"Got format: {gpu_raw}"
        )

    # Step 2: Parse the entities into a list of tuples containing 
    # the gpu type, count, and memory
    individual_gpus = []
    for gpu_type_raw, gpu_mem_raw in zip(entities[:T], entities[T:]):
        
        # 2a. Parse the gpu name and count using regex
        gres_regex = r"(\w*):(\d)(\(?.*\)?)"
        gres_match = re.match(gres_regex, gpu_type_raw)
        gpu_count = int(gres_match.groups()[1])
        gpu_type1 = gres_match.groups()[0]

        # 2b. Parse the memory using regular split
        gpu_type2, _, gmem = gpu_mem_raw.split(":")

        # 2c. Check if the gpu type is the same
        if not gpu_type2 == gpu_type1:
            raise ValueError(
                f"Parsing error @{node_name}: GPU type mismatch {gpu_type1} != {gpu_type2}"
            )

        # 2d. Append the gpu type, count, and memory to the list
        individual_gpus.append((gpu_type1, gpu_count, gmem))

    # Step 3: Sum the gpu counts and format the gpu type
    gpu_count = sum([count for _, count, _ in individual_gpus])
    gpu_type = ",".join([f"{y}x{x}({m})" for x, y, m in individual_gpus])
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

            try:
                gpu_count, gpu_type = parse_complex_gpu_descriptions(
                    gpu_raw, node_name
                )
            except ValueError:
                # print(f"Could not parse gpu info: {gpu_raw}")
                print(f"Could not parse gpu info for {node_name}: {gpu_raw}")
                # continue
                gpu_count, gpu_type = 0, "unknown"

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


