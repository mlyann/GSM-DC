import os, re, torch, random, math, copy, base64
import json
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
from typing import Any, List, Union, Tuple, Dict
import hashlib
from const.params import mod
import numpy as np
import networkx as nx
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def to_sketch(problem, prob: str=None, sol: str=None):
    sketch = {}
    replace_inst = [problem.get_ntn(param) for param in problem.all_param if param[0] == 0]
    replace_inter = [problem.get_ntn(param) for param in problem.all_param if param[0] == 1]
    if prob:
        # replace the problem parameters
        for param_str in replace_inst:
            prob = prob.replace(param_str, "Inst")
        for param_str in replace_inter:
            prob = prob.replace(param_str, "Inter")
        # replace the numbers
        for i in range(mod-1, 0, -1):
            prob = prob.replace(str(i), "0")
        # replace the question parameters
        for Name in problem.N:
            for n in Name:
                prob = prob.replace(n, "Item")
        for ln in problem.ln:
            prob = prob.replace(ln, "Categ")
        # print(f"prob:\n{prob}")
        sketch['prob'] = prob
    if sol:
        for param_str in replace_inst:
            sol = sol.replace(param_str, "Inst")
        for param_str in replace_inter:
            sol = sol.replace(param_str, "Inter")
        # replace the numbers
        for i in range(mod-1, 0, -1):
            sol = sol.replace(str(i), "0")

        # replace the symbols
        sol = sol.replace(";", ".")
        solution_sentences = sol.split(". ")
        solution_grouped_parts = [solution_sentence.split(" ") for solution_sentence in solution_sentences]
        ntns = []
        for solution_group in solution_grouped_parts:
            for part in solution_group:
                if part in problem.all_symbols:
                    if part not in ntns:
                        ntns.append(part)
        ntn_dict = {}
        for i, ntn in enumerate(ntns):
            ntn_dict[ntn] = problem.all_symbols[i]
        
        for i, solution_group in enumerate(solution_grouped_parts):
            for j in range(len(solution_group)):
                part = solution_group[j]
                if part in problem.all_symbols:
                    solution_grouped_parts[i][j] = ntn_dict[part]
        
        solution_sentences = [" ".join(solution_group) for solution_group in solution_grouped_parts]
        sol = ". ".join(solution_sentences)
        # print(f"sol:\n{sol}")
        sketch['sol'] = sol
    return sketch

def to_hash(hash_string: str, mod_num=mod):
    '''
    return a hash value in [0, 1, ..., mod-1]
    use after self.to_problem()
    '''
    hash_object = hashlib.sha256(hash_string.encode())
    # Get the hexadecimal representation and convert it to an integer
    hash_integer = int(hash_object.hexdigest(), 16)
    # Return the hash value modulo a+1
    return hash_integer % (mod_num)

def choose_from_softmax(lst: list, weight: list):
    weight = np.array(weight)
    e_x = np.exp(weight - np.max(weight))
    p = e_x / np.sum(e_x)
    return np.random.choice(lst, p=p)

def show_info(output: List[int], problem, req_return=False):
    print()
    problem.display()
    pre = 0
    for i in output:
        if i == 222:
            break
        pre += 1
    
    dash = "-"

    if 223 not in output or 224 not in output:
        print(f"223 ({223 in output}), 224 ({224 in output}).")
        prob_text = " " + ". ".join(problem.problem) + "." # f" Answer in detail level_{problem.detail_level_}."
        print(f"Problem: {dash*32}\n{prob_text}")
        print(f"GPT Output:", tokenizer.decode(output[pre:], skip_special_tokens=True))
        print(f"Correct Solution: {dash*32}\n{' ' + '. '.join(problem.solution) + '.'}")
        return
    index_223 = output.index(223)
    index_224 = output.index(224)
    # split the list into three parts
    prob_token = output[:index_223]
    sol_token = output[index_223+1:index_224]
    ans_token = output[index_224+1:]

    prob_text = tokenizer.decode(prob_token, skip_special_tokens=True)
    sol_text = tokenizer.decode(sol_token, skip_special_tokens=True)
    ans_text = tokenizer.decode(ans_token, skip_special_tokens=True)

    print(f"Problem: {dash*32}\n{prob_text}")
    print(f"GPT Solution: {dash*32}\n{sol_text}")
    print(f"Correct Solution: {dash*32}\n{' ' + '. '.join(problem.solution) + '.'}")
    notation = "=" if ans_text == " " + str(problem.ans) else "!="
    print(f"Answer: {dash*32}\nGPT ans{ans_text} {notation} Correct ans {problem.ans}")

    if req_return:
        return sol_text, ans_text

def decode_detail_level(detail_level):
    if detail_level == None:
        return None
    case0 = detail_level % 3
    rand_ = detail_level // 3
    case1 = rand_ % 2
    rand_ = rand_ // 2
    case2 = rand_
    '''if case0 == 0:
        define_var = True
        define_detail = True
    elif case0 == 1:
        define_var = True
        define_detail = False
    else:
        define_var = False
        define_detail = False
    name_omit = True if case1 == 1 else False
    cal_omit = True if case2 == 1 else False'''
    return f"{case0}{case1}{case2}"

def seed_from_list(lst):
    # Convert the list of integers into a concatenated string
    concatenated = '-'.join(map(str, lst))

    # Hash the string using SHA-256
    hashed = hashlib.sha256(concatenated.encode())

    # Convert the hash into an integer seed value
    seed = int(hashed.hexdigest(), 16) % (2**32 - 1)  # Limit to the size of a 32-bit integer

    return seed

def basic_collate_fn(examples):
    transposed = tuple(zip(*examples))
    return transposed
    # data_lst = []
    # problem_lst = []
    # for data, problem in examples:
    #     data_lst.append(data)
    #     problem_lst.append(problem)
    # return data_lst, problem_lst

def is_float(number: str):
    if number.count(".") == 1:
        if number.replace(".", "").isdigit():
            return True
    return False

def idle_func(*args):
    pass

def display_table(table: dict):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    def round_elems(x):
        if isinstance(x, float):
            return round(x, 2)
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, tuple) or isinstance(x, list):
            return tuple(round_elems(item) for item in x)
        return x
    df = pd.DataFrame.from_dict(data=table, orient='index').transpose()
    df = df.map(round_elems)
    print(df)
    return

    num_format = "{:.2f}"
    none_representation = "None"
    column_widths = {
        key: max(
            len(num_format.format(val)) if val is not None else len(none_representation)
            for val in value
        )
        for key, value in table.items()
    }

    # Adjust column widths to also accommodate the header lengths
    header_widths = {key: len(key) for key in table.keys()}
    max_widths = {key: max(header_widths[key], column_widths[key]) for key in table.keys()}

    # Print the header
    header = " | ".join(f"{key:{max_widths[key]}}" for key in table.keys())
    print(header)
    print("-" * len(header))

    # Print each row
    for i in range(len(next(iter(table.values())))):  # Assuming all lists are of the same length
        row = " | ".join(
            f"{num_format.format(table[key][i]):{max_widths[key]}}" if table[key][i] is not None else f"{none_representation:{max_widths[key]}}"
            for key in table.keys()
        )
        print(row)

def random_topological_sort(graph: nx.DiGraph):
    # Make sure it's a directed acyclic graph
    if not nx.is_directed_acyclic_graph(graph):
        return None

    # Step 1: Compute the in-degree for each vertex
    in_degree = dict(graph.in_degree())

    # Step 1: Add vertices with in-degree 0 to the set
    zero_in_degree = [v for v, d in in_degree.items() if d == 0]

    # Step 2: While the set is not empty
    topological_order = []
    while zero_in_degree:
        # Step 2a: Randomly remove a vertex u from the set
        u = random.choice(zero_in_degree)
        zero_in_degree.remove(u)

        # Step 2b: Add u to the topological order
        topological_order.append(u)

        # Step 2c: For each vertex v adjacent to u, decrease in-degree and check if it becomes 0
        for v in graph.successors(u):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                zero_in_degree.append(v)

    return topological_order

def subgraph_with_paths_to_node(G: nx.DiGraph, target_node):
    # Step 1: Reverse the graph
    reversed_G = G.reverse()

    # Step 2: Perform BFS or DFS to find nodes with a path to 'target_node'
    reachable_nodes = set(nx.single_source_shortest_path(reversed_G, target_node))

    # Step 3: Create a subgraph with these nodes
    subgraph = G.subgraph(reachable_nodes)

    return subgraph

def wrap_label(label:str, max_width=10):
    new_label = ""
    i = 0
    for pos, l in enumerate(label):
        if i == max_width:
            if l == " ":
                new_label += "\n"
                i = 0
                continue
            elif new_label[-1] == " ":
                new_label = new_label[:-1] + "\n" + l
                i = 1
            elif new_label[-2] == " ":
                new_label = new_label[:-2] + "\n" + new_label[-1] + l
                i = 2
            elif new_label[-3] == " ":
                new_label = new_label[:-3] + "\n" + new_label[-2:] + l
                i = 3
            else:
                new_label = new_label[:-1] + "-\n" + new_label[-1]
                new_label += l
                i = 2
        else:
            new_label += l
            i += 1
    #print(f"Label conert: {label} --> {new_label}")
    return new_label

def mask_label(label: List[int]):
    length = len(label)
    label = copy.copy(label)
    mask = False
    for i in range(length-1, -1, -1):
        if label[i] == 223:
            break
        if not mask:
            if label[i] == 28767: # BACK
                mask = True
        else: # if in mask status
            if label[i] == 355: # as
                pass
            elif label[i] == 500 and label[i-1] == 2896: # Define
                mask = False
                pass
            else:
                label[i] = -1
    
    return label

def mask_label2(label: List[int]):
    length = len(label)
    label_new = copy.copy(label)
    mask = False
    for i in range(length-1, -1, -1):
        if label[i] == 223:
            break
        if not mask:
            if label[i] == 28767: # BACK
                mask = True
        else: # if in mask status
            if label[i+1] == 500 and label[i] == 2896: # Define
                mask = False
            label_new[i] = -1
    
    return label_new

def shortest_path_lengths(G: nx.DiGraph) -> Dict[List[int], Dict[List[int], int]]:
    # Compute shortest path lengths from each node
    all_paths = dict(nx.all_pairs_shortest_path_length(G))

    dist_dict = {}
    for i in G.nodes():
        dist_dict[i] = {}
        for j in G.nodes():
            # Distance from a node to itself is 0
            if i == j:
                dist_dict[i][j] = 0
            # Check if j is reachable from i
            elif j in all_paths[i]:
                dist_dict[i][j] = all_paths[i][j]
            else:
                dist_dict[i][j] = -1

    return dist_dict

def hash_string(input_string: str) -> bytes:
    # Create a new SHA-256 hash object
    hasher = hashlib.sha256()
    # Update the hash object with the bytes of the input string
    hasher.update(input_string.encode('utf-8'))
    # Return the digest of strings passed to the update method so far
    return hasher.digest()

def hash_str2str(input_string: str) -> str:
    '''
    A deterministic method to hash a string into another string with length <= 44
    '''
    # Hash the input string
    hash_bytes = hash_string(input_string)
    # Encode the bytes into a Base64 string to reduce length
    base64_bytes = base64.urlsafe_b64encode(hash_bytes)
    # Convert bytes to string
    return base64_bytes.decode('utf-8')

class MyPrint(object):
    def __init__(self) -> None:
        self.txt: list[tuple] = []
        self.temp_txt: list[tuple] = []
    
    def __call__(self, *args) -> None:
        self.txt.append(args)
        self.temp_txt.append(args)
    
    def display(self) -> None:
        for args in self.txt:
            print(*args)
        
        self.temp_txt = []

    @property
    def string(self):
        return "\n".join([" ".join(str(part_txt) for part_txt in part) for part in self.txt])

    def save(self, path: str):
        pass




