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
from typing import List, Tuple, Optional
import re
import subprocess
import warnings

# Constants
DOWN_STATES = ["down", "drain", "fail", "fail*", "down*", "draining", "drained"]
PUBLIC_PARTITIONS = ["asteroids", "asteroids*", "universe", "universe*"]

# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"


def color_print(text: str, color: str) -> str:
    """
    Add color to text for terminal output.
    
    Args:
        text (str): Text to color
        color (str): Color code from Colors class
        
    Returns:
        str: Colored text with reset code
        
    Example:
        >>> print(color_print("Success", Colors.GREEN))
        Success  # Will be green in terminal
    """
    return f"{color}{text}{Colors.RESET}"


@dataclass
class GPUTypeStruct:
    """
    Represents the structure of GPU types on a node.
    
    Attributes:
        count (int): Number of GPUs of this type
        memory_gb (int): Memory in GB for each GPU
        type (str): GPU type name (e.g. "A100")
    """
    count: int
    memory_gb: int
    type: str


class NodeParseError(Exception):
    """Raised when there is an error parsing node information from sinfo output."""

    pass


class GPUParseError(Exception):
    """Raised when there is an error parsing GPU information from sinfo output."""

    pass


@dataclass
class Node:
    """
    Represents a compute node in the cluster.

    Attributes:
        name (str): Name of the node
        gpu_type (str): Type and configuration of GPUs (e.g. "2xA100(80G)")
        gpu_count (int): Total number of GPUs on the node
        state (str): Current state of the node (e.g. "idle", "allocated")
        partition (str): SLURM partition the node belongs to
        is_asteroid (bool): Whether the node is in the asteroids partition
        is_unavailable (bool): Whether the node is unavailable (down/drained/private)
        is_private (bool): Whether the node is private (not in universe/asteroids)
        gpu_types (List[GPUTypeStruct]): List of GPU types with their counts and memory

    Example:
        >>> node_raw = "node1||idle||asteroids||gpu:A100:2(S:0-15),gpumem:A100:no_consume:80G"
        >>> node = Node(node_raw)
        >>> print(node)
        Name: node1                    GPUs: 2(A100(80G))                    State: idle                 asteroid   public
    """

    name: str
    gpu_type: str
    gpu_count: int
    state: str
    partition: str
    is_asteroid: bool
    is_private: bool
    is_unavailable: bool
    gpu_types: List[GPUTypeStruct]

    def __init__(self, node_raw: str) -> None:
        """
        Initialize a Node from raw sinfo output.

        Args:
            node_raw (str): Raw string from sinfo containing node information
                Format: "node_name||status||partition||gpu_info"

        Raises:
            NodeParseError: If the node information cannot be parsed
            GPUParseError: If the GPU information cannot be parsed
        """
        try:
            node_name, status, partition, gpu_raw = [
                x for x in node_raw.split("||") if x
            ]
        except Exception as e:
            raise NodeParseError(
                f"Could not parse node info from: {node_raw}. Expected format: node_name||status||partition||gpu_info"
            )

        try:
            gpu_count, gpu_type, gpu_types = parse_complex_gpu_descriptions(
                gpu_raw, node_name
            )
        except GPUParseError as e:
            warnings.warn(
                f"Could not parse gpu info for {node_name}: {gpu_raw}. Using default values."
            )
            gpu_count, gpu_type, gpu_types = 0, "unknown", []

        is_private = partition not in PUBLIC_PARTITIONS

        self.name = node_name
        self.gpu_type = gpu_type
        self.gpu_count = int(gpu_count)
        self.state = status
        self.partition = partition
        self.is_asteroid = partition == "asteroids"
        self.is_private = is_private
        self.is_unavailable = status in DOWN_STATES or is_private
        self.gpu_types = gpu_types

    def __repr__(self) -> str:
        """Returns a formatted string representation of the node."""
        # Create compact GPU description
        gpu_desc = []
        for gpu in self.gpu_types:
            gpu_desc.append(f"{gpu.count}x{gpu.memory_gb}GB")
        gpu_desc_str = ",".join(gpu_desc) if gpu_desc else "unknown"
        
        return (
            f"Name: {self.name}".ljust(25)
            + f"GPUs: {gpu_desc_str}".ljust(25)
            + f"State: {self.state}".ljust(20)
            + ("asteroid" if self.is_asteroid else "universe").ljust(10)
            + ("private" if self.is_private else "public").ljust(10)
        )


def convert_to_gb(memory_str: str) -> str:
    """
    Convert memory string to GB, rounding down to the nearest integer.

    Args:
        memory_str (str): Memory string in format like "80G" or "46068M"

    Returns:
        str: Memory string in GB (e.g. "80G")

    Example:
        >>> convert_to_gb("80G")
        "80G"
        >>> convert_to_gb("46068M")
        "46G"
    """
    if memory_str.endswith("G"):
        return memory_str
    elif memory_str.endswith("M"):
        # Convert MB to GB and round down
        mb = int(memory_str[:-1])
        gb = mb // 1024
        return f"{gb}G"
    else:
        raise ValueError(
            f"Unsupported memory unit in {memory_str}. Expected 'G' or 'M'."
        )


def convert_to_gb_int(memory_str: str) -> int:
    """
    Convert memory string to integer GB, rounding down to the nearest integer.

    Args:
        memory_str (str): Memory string in format like "80G" or "46068M"

    Returns:
        int: Memory in GB as integer

    Example:
        >>> convert_to_gb_int("80G")
        80
        >>> convert_to_gb_int("46068M")
        46
    """
    if memory_str.endswith("G"):
        return int(memory_str[:-1])
    elif memory_str.endswith("M"):
        # Convert MB to GB and round down
        mb = int(memory_str[:-1])
        return mb // 1024
    else:
        raise ValueError(
            f"Unsupported memory unit in {memory_str}. Expected 'G' or 'M'."
        )


def parse_complex_gpu_descriptions(
    gpu_raw: str, node_name: str
) -> Tuple[int, str, List[GPUTypeStruct]]:
    """
    Parses GPU info from sinfo output. Can handle nodes that contain multiple GPU types.

    Args:
        gpu_raw (str): Raw string from sinfo containing GPU info
            Format: 'gpu:name1:2(S:0-15),gpu:name2:2(S:0-15),gpumem:name1:no_consume:48G,gpumem:name2:no_consume:48G'
        node_name (str): Name of the node for error reporting

    Raises:
        GPUParseError: If the GPU info does not match the expected format

    Returns:
        Tuple[int, str, List[GPUTypeStruct]]: Total number of GPUs, their type description,
            and a list of GPU type structures

    Example:
        >>> gpu_raw = "gpu:A100:2(S:0-15),gpumem:A100:no_consume:80G"
        >>> count, type, types = parse_complex_gpu_descriptions(gpu_raw, "node1")
        >>> print(f"{count}x{type}")
        2xA100(80G)
    """
    try:
        # Step 1: Parse the gpu_raw string into a list of entities
        entities = re.split(r"gpu:|gpumem:", gpu_raw)[1:]
        entities = [e[:-1] if e[-1] == "," else e for e in entities]

        if len(entities) % 2 == 1:
            raise GPUParseError(
                f"@{node_name}: Odd number of entities: {len(entities)}"
            )

        T = len(entities) // 2
        recon_gpu_raw = ",".join(
            ["gpu:" + e for e in entities[:T]]
            + ["gpumem:" + e for e in entities[T:]]
        )
        if recon_gpu_raw != gpu_raw:
            raise GPUParseError(
                f"@{node_name}: Expected format: {recon_gpu_raw}; got format: {gpu_raw}"
            )

        # Step 2: Parse the entities into a list of tuples
        individual_gpus: List[Tuple[str, int, str]] = []
        gpu_types: List[GPUTypeStruct] = []
        
        for gpu_type_raw, gpu_mem_raw in zip(entities[:T], entities[T:]):
            gres_regex = r"(\w*):(\d)(\(?.*\)?)"
            gres_match = re.match(gres_regex, gpu_type_raw)
            if not gres_match:
                raise GPUParseError(
                    f"@{node_name}: Could not parse GPU type: {gpu_type_raw}"
                )

            gpu_count = int(gres_match.groups()[1])
            gpu_type1 = gres_match.groups()[0]

            try:
                gpu_type2, _, gmem = gpu_mem_raw.split(":")
                # Convert memory to GB
                gmem = convert_to_gb(gmem)
                gmem_int = convert_to_gb_int(gmem)
            except (ValueError, IndexError) as e:
                raise GPUParseError(
                    f"@{node_name}: Could not parse GPU memory: {gpu_mem_raw}. Error: {e}"
                )

            if not gpu_type2 == gpu_type1:
                raise GPUParseError(
                    f"@{node_name}: GPU type mismatch {gpu_type1} != {gpu_type2}"
                )

            individual_gpus.append((gpu_type1, gpu_count, gmem))
            gpu_types.append(GPUTypeStruct(count=gpu_count, memory_gb=gmem_int, type=gpu_type1))

        # Step 3: Sum the gpu counts and format the gpu type
        gpu_count = sum(count for _, count, _ in individual_gpus)
        gpu_type = ",".join(f"{y}x{x}({m})" for x, y, m in individual_gpus)
        return gpu_count, gpu_type, gpu_types
        
    except GPUParseError:
        return 0, "unknown", []


class NodeMaster:
    """
    Manages a collection of compute nodes in the cluster.

    This class fetches node information from SLURM's sinfo command and provides
    methods to access and filter nodes based on various criteria.
    """

    def __init__(self) -> None:
        """
        Initialize NodeMaster by fetching current node information from SLURM.

        Raises:
            subprocess.CalledProcessError: If sinfo command fails
        """
        sinfo_output = (
            subprocess.check_output(["sinfo", "-N", "-o", "%N||%T||%P||%G"])
            .decode("utf-8")
            .strip()
            .split("\n")[1:]
        )
        nodes = []
        for node_raw in sinfo_output:
            try:
                nodes.append(Node(node_raw))
            except NodeParseError as e:
                warnings.warn(
                    f"Could not parse node info: {node_raw}. Error: {e}"
                )
                continue

        self.nodes = sorted(
            nodes,
            key=lambda x: (
                x.is_unavailable,  # Available nodes first
                -x.gpu_count,  # Sort by gpu count in descending order
                x.gpu_type,  # Sort by GPU type in ascending order
                x.name,  # Sort by node name in ascending order
            ),
        )
        self.total_gpu_count = sum(node.gpu_count for node in self.nodes)
        self.total_gpu_count_available = sum(
            node.gpu_count for node in self.nodes if not node.is_unavailable
        )
        self.total_non_asteroid_count = sum(
            node.gpu_count
            for node in self.nodes
            if not node.is_asteroid and not node.is_unavailable
        )

    def __repr__(self) -> str:
        """Returns a formatted string representation of the cluster state."""
        ret = f"Total GPUs: {self.total_gpu_count}\n"
        ret += (
            f"Total GPUs currently online: {self.total_gpu_count_available}\n"
        )
        ret += f"Total non-asteroid GPUs: {self.total_non_asteroid_count}\n"
        ret += "\nNodes:\n"
        for node in self.nodes:
            node_str = str(node)
            if node.is_unavailable:
                ret += color_print(node_str, Colors.RED) + "\n"
            else:
                ret += color_print(node_str, Colors.GREEN) + "\n"
        return ret


if __name__ == "__main__":
    nodes = NodeMaster()
    print(nodes)
