# ------------------------------------------------------------------------------
# File: nodes.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Mon Aug 12 2024
#
# Parses sinfo and returns a list of nodes with their GPU type and count.
#
# ------------------------------------------------------------------------------
# %%
import subprocess
down_states = ['down', 'drain', 'fail', 'fail*', 'down*']

class Node:
    def __init__(self, raw_string):
        self.name, gres, self.state = [x for x in raw_string.split() if x]
        self.gpu_type, self.gpu_count = self.parse_gres(gres)
        self.is_down = self.state in down_states

    def __repr__(self):
        return \
            f'Name: {self.name}'.ljust(20) +\
            f'GPUs:{self.gpu_count}x{self.gpu_type}'.ljust(15) +\
            f'State:{self.state}'.ljust(15)
    
    def parse_gres(self, gres: str):
        res_type, gpu_type, number = gres.split(':')
        if res_type == 'gpu':
            return gpu_type, int(number)
        else:
            raise NotImplementedError('Only GPU node types are supported.')


class NodeMaster():
    def __init__(self) -> None:
        self.nodes = self.read_all_nodes()
        self.nodes = sorted(self.nodes, key=lambda x: x.gpu_type)
        # self.down_states = ['down', 'drain', 'fail', 'fail*', 'down*']
        self.total_gpu_count = sum([node.gpu_count for node in self.nodes])
        self.total_gpu_count_available = sum(
            [node.gpu_count for node in self.nodes if not node.is_down])

    def read_all_nodes(self):
        output = subprocess.check_output([
            'sinfo', '-N', '-O', 'NodeList:15,Gres:15,StateLong:6'])
        output = output.decode('utf-8')
        nodes = output.split('\n')[1:]
        nodes = [Node(node) for node in nodes if node]
        return nodes
    
    def __repr__(self):
        ret = f'Total GPUs: {self.total_gpu_count}\n'
        ret += f'Currently available GPUs: {self.total_gpu_count_available}\n'
        ret += '\nNodes:\n'
        for node in self.nodes:
            ret += str(node) + '\n'
        return ret

if __name__ == '__main__':
    nodes = NodeMaster()
    print(nodes)
