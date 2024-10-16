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
down_states = ['down', 'drain', 'fail', 'fail*', 'down*', 'draining', 'drained']

@dataclass
class Node:
    name: str
    gpu_type: str
    gpu_count: int
    state: str
    is_down: bool
    partition: str
    is_asteroid: bool

    def __repr__(self):
        return \
            f'Name: {self.name}'.ljust(20) +\
            f'GPUs: {self.gpu_count}x{self.gpu_type}'.ljust(20) +\
            f'State: {self.state}'.ljust(20) +\
            ('asteroid' if self.is_asteroid else 'universe')
            # f'Partition: {self.partition}'.ljust(15)


class NodeMaster():
    def __init__(self) -> None:
        gres_regex = r'gpu:(\w*):(\d)(\(?.*\)?)'
        sinfo_output = subprocess.check_output([
            'sinfo', '-N', '-O', 'NodeList,Gres,StateLong,Partition']
            # 'sinfo', '-N', '-O', 'NodeList,Gres,StateLong']
            ).decode('utf-8').strip().split('\n')[1:]
        nodes = []
        for node_raw in sinfo_output:
            node_name, gpu_raw, status, partition = [x for x in node_raw.split() if x]
            matches = re.match(gres_regex, gpu_raw)
            if not matches:
                raise NotImplementedError(\
                    f'Could not parse GPU info for node {node_name}: ' +\
                        f'{gpu_raw}' 'Please check the sinfo output.')
            else:
                gpu_type, gpu_count, _ = matches.groups()
            nodes.append(Node(node_name, gpu_type, int(gpu_count), 
                              status, status in down_states, 
                              partition, partition=='asteroids'))

        self.nodes = sorted(nodes, key=lambda x: (
            # Cluster nodes first, asteroids last
            -x.gpu_count,  # Sort by gpu count in descending order
            x.is_down,  # Available nodes first
            x.gpu_type,  # Sort by GPU type in ascending order
            x.name  # Sort by node name in ascending order
        ))
        self.total_gpu_count = sum([node.gpu_count for node in self.nodes])
        self.total_gpu_count_available = sum(
            [node.gpu_count for node in self.nodes if not node.is_down])
    
    def __repr__(self):
        ret = f'Total GPUs: {self.total_gpu_count}\n'
        ret += f'Total GPUs currently online: {self.total_gpu_count_available}\n'
        ret += '\nNodes:\n'
        for node in self.nodes:
            ret += str(node) + '\n'
        return ret

if __name__ == '__main__':
    nodes = NodeMaster()
    print(nodes)