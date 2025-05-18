import json
import networkx as nx
import numpy as np
import random, copy, math, hashlib, string
from heapq import heappush, heappop
from itertools import count, product
from typing import List, Dict, Union, Callable, Any
from transformers import AutoTokenizer
from const.params import mod, try_num, feasible_symbols
from data_gen.prototype.id_gen import IdGen_PT
from math_gen.problem_gen import Problem,Expression
from tools.tools_test import true_correct
from tools.tools import tokenizer

def save_problem_to_json(problem: Problem, file_path: str) -> None:
    data_dict = problem_to_json(problem)
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2)

#    Convert a Problem object to JSON 
#    (including topological_order if it exists, 
#     and also storing whole_template).
def expression_to_relationship(expr: Expression) -> dict:
    """Converts an expression into a relationship dictionary.
    
    Args:
        expr (Expression): The expression object to be converted.
    
    Returns:
        dict: A dictionary representation of the expression, containing the operation,
              its value (if applicable), and a list of child expressions.
    """
    rel = {
        "op": expr.op if expr.op is not None else "const",
        "children": []
    }
    if hasattr(expr, "value"):
        rel["value"] = expr.value.a
    for child in expr.param_list:
        rel["children"].append(expression_to_relationship(child))
    return rel

def problem_to_json(problem: Problem) -> dict:
    d  = int(problem.d)
    w0 = int(problem.w0)
    w1 = int(problem.w1)
    e  = int(problem.e)
    p  = float(problem.p)

    # question_index
    if hasattr(problem, "ques_idx") and problem.ques_idx is not None:
        question_index = [int(x) for x in problem.ques_idx]
    else:
        question_index = None

    # final_answer
    final_answer = problem.ans if hasattr(problem, "ans") else None
    if final_answer is not None:
        final_answer = int(final_answer)

    # --- 修改 node_data: 使用节点的 label 作为 key, 并保存节点坐标和 unique ---
    node_data = {}
    for (layer, idx), data in problem.graph.nodes(data=True):
        label = str(problem.N[layer][idx])
        node_data[label] = {
            "node": f"({int(layer)}, {int(idx)})",
            "unique": bool(data.get('unique', False))
        }

    # edges in structure graph
    edges = []
    for u, v in problem.graph.edges():
        edges.append([str(u), str(v)])

    # template edges
    template_edges = []
    for u, v in problem.template.edges():
        template_edges.append([str(u), str(v)])

    # whole_template edges
    whole_template_edges = []
    if hasattr(problem, "whole_template"):
        for u, v in problem.whole_template.edges():
            whole_template_edges.append([str(u), str(v)])

    # problem_text, solution_text
    problem_text = problem.problem if hasattr(problem, "problem") else []
    solution_text = problem.solution if hasattr(problem, "solution") else []

    # ln
    ln = [str(x) for x in problem.ln]

    # all_param
    all_param = []
    if hasattr(problem, "all_param"):
        all_param = [str(param) for param in problem.all_param]

    topo_list = []
    if hasattr(problem, "topological_order") and problem.topological_order:
        for param in problem.topological_order:
            param_str = str(param)
            # param => (l, i, j, k)
            l, i, j, k = param
            if l == -1:
                description = "RNG"
            elif l == 0:
                name0 = problem.N[i][j]
                name1 = problem.N[i+1][k]
                description = f"{name0}{problem.args['dot']}{name1}"
            elif l == 1:
                name0 = problem.N[i][j]
                cat   = problem.ln[k]
                description = f"{name0}{problem.args['dot']}{cat}"
            else:
                description = f"UnsupportedParam{param}"
            topo_list.append({
                "param": param_str,
                "description": description
            })

    # keep everything matching with json
    expression_relationships = {}
    if hasattr(problem, "sketch"):
        for param, expr in problem.sketch.items():
            expression_relationships[str(param)] = expression_to_relationship(expr)

    out_dict = {
        "problem_info": {
            "d": d,
            "w0": w0,
            "w1": w1,
            "e": e,
            "p": p,
            "final_answer": final_answer,
            "question_index": question_index
        },
        "node_data": node_data,
        "edges": edges,
        "template_edges": template_edges,
        "whole_template_edges": whole_template_edges,
        "ln": ln,
        "all_param": all_param,
        "problem_text": problem_text,
        "solution_text": solution_text,
        "topological_order": topo_list,
        "expression_relationships": expression_relationships
    }
    return out_dict

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# 2) Load from JSON and re-construct a Problem object,
#    then restore topological_order, whole_template, and compute n_op.
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def load_problem_from_json(file_path: str) -> dict:
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def parse_param_str(s: str):
    inside = s.strip("()")
    parts = inside.split(",")
    if len(parts) != 4:
        raise ValueError(f"param tuple must have length 4, got {s}")
    return tuple(int(x.strip()) for x in parts)

def rebuild_problem_from_json(data_dict: dict) -> Problem:
    info = data_dict["problem_info"]
    d  = int(info["d"])
    w0 = int(info["w0"])
    w1 = int(info["w1"])
    e  = int(info["e"])
    p  = float(info["p"])
    final_answer = info["final_answer"]
    if final_answer is not None:
        final_answer = int(final_answer)
    question_index = info["question_index"]

    # IMPORTANT: Hardcode the dot as "'s "
    args = {
        "rand_perm": "none",
        "define_var": True,
        "define_detail": True,
        "inter_var": True,
        "name_omit": False,
        "cal_omit": False,
        "dot": "'s ",
        "symbol_method": "rand",
        "sol_sort": False,
        "perm": False
    }
    problem = Problem(d, w0, w1, e, p, args=args)

    # change node, reconstruct
    node_data = data_dict["node_data"]
    layer_counts = {}
    for key, value in node_data.items():
        node_str = value["node"]  # eg "(0, 0)"
        layer_str, idx_str = node_str.strip("()").split(",")
        layer = int(layer_str)
        idx   = int(idx_str)
        layer_counts[layer] = max(layer_counts.get(layer, 0), idx+1)
    for i in range(d):
        problem.l[i] = layer_counts.get(i, 0)

    problem.graph = nx.DiGraph()
    for i in range(d):
        for j in range(problem.l[i]):
            problem.graph.add_node((i, j), unique=False)

    problem.N = []
    for i in range(d):
        problem.N.append([""] * problem.l[i])

    for key, value in node_data.items():
        node_str = value["node"]
        layer_str, idx_str = node_str.strip("()").split(",")
        layer = int(layer_str)
        idx   = int(idx_str)
        label = key  # 使用 JSON 中的键作为 label
        problem.N[layer][idx] = label
        problem.graph.nodes[(layer, idx)]["unique"] = bool(value["unique"])
        if bool(value["unique"]):
            problem.unique.append((layer, idx))

    edges = data_dict["edges"]
    for u_str, v_str in edges:
        ulayer_idx = u_str.strip("()").split(",")
        u_layer = int(ulayer_idx[0])
        u_idx   = int(ulayer_idx[1])
        vlayer_idx = v_str.strip("()").split(",")
        v_layer = int(vlayer_idx[0])
        v_idx   = int(vlayer_idx[1])
        problem.graph.add_edge((u_layer, u_idx), (v_layer, v_idx), chosen=False)

    problem.G = []
    for i in range(d - 1):
        M = np.zeros((problem.l[i], problem.l[i+1]), dtype=bool)
        for j in range(problem.l[i]):
            for k in range(problem.l[i+1]):
                if problem.graph.has_edge((i, j), (i+1, k)):
                    M[j, k] = True
        problem.G.append(M)

    ln_list = data_dict.get("ln", [])
    problem.ln = [str(x) for x in ln_list]

    # template
    problem.template = nx.DiGraph()
    template_edges = data_dict["template_edges"]
    node_set = set()
    for u_str, v_str in template_edges:
        node_set.add(u_str)
        node_set.add(v_str)
    for n_str in node_set:
        param_tup = parse_param_str(n_str)
        problem.template.add_node(param_tup)
    for u_str, v_str in template_edges:
        u_tup = parse_param_str(u_str)
        v_tup = parse_param_str(v_str)
        problem.template.add_edge(u_tup, v_tup)

    # whole_template
    whole_template_edges = data_dict.get("whole_template_edges", [])
    problem.whole_template = nx.DiGraph()
    wt_node_set = set()
    for u_str, v_str in whole_template_edges:
        wt_node_set.add(u_str)
        wt_node_set.add(v_str)
    for n_str in wt_node_set:
        param_tup = parse_param_str(n_str)
        problem.whole_template.add_node(param_tup)
    for u_str, v_str in whole_template_edges:
        u_tup = parse_param_str(u_str)
        v_tup = parse_param_str(v_str)
        problem.whole_template.add_edge(u_tup, v_tup)

    # all_param
    problem.all_param = []
    if "all_param" in data_dict:
        for sp in data_dict["all_param"]:
            problem.all_param.append(parse_param_str(sp))

    problem.ans = final_answer if final_answer is not None else 0
    if question_index is not None:
        problem.ques_idx = tuple(question_index)

    # topological_order
    topo_list = data_dict.get("topological_order", [])
    if topo_list:
        problem.topological_order = [
            parse_param_str(item["param"]) if isinstance(item, dict) else parse_param_str(item)
            for item in topo_list
        ]
    else:
        problem.topological_order = []

    # Recompute n_op from topological_order
    n_op = 0
    for param in problem.topological_order:
        num_pre = len(list(problem.template.predecessors(param)))
        if num_pre <= 2:
            n_op += 1
        else:
            n_op += num_pre - 1
    problem.n_op = n_op

    build_name2param_dict(problem)

    # Also restore textual problem/solution
    prob_text = data_dict.get("problem_text", [])
    sol_text  = data_dict.get("solution_text", [])
    problem.problem  = prob_text
    problem.solution = sol_text

    return problem

def build_name2param_dict(problem: Problem):
    problem.name2param_dict = {}
    for param in problem.all_param:
        l, i, j, k = param
        if l == -1:
            param_name = "RNG"
        elif l == 0:
            name0 = problem.N[i][j]
            name1 = problem.N[i+1][k]
            param_name = f"{name0}{problem.args['dot']}{name1}"
        elif l == 1:
            name0 = problem.N[i][j]
            cat   = problem.ln[k]
            param_name = f"{name0}{problem.args['dot']}{cat}"
        else:
            param_name = f"UnsupportedParam{param}"
        problem.name2param_dict[param_name] = param

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Example
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def example_demo(op =3):
    from data_gen.prototype.id_gen import IdGen_PT
    ava_hash = [i for i in range(9999)]
    id_gen = IdGen_PT(
        style="light",
        op_style="light",
        op=op,
        perm_level=5,
        detail_level=0
    )

    # p_format="pq" => problem text + final question
    id_gen.gen_prob(ava_hash=ava_hash, p_format="pq")
    save_problem_to_json(id_gen.problem, "my_problem.json")
    print("Saved problem to my_problem.json")

def run_validation():
    data_dict = load_problem_from_json("my_problem.json")
    problem = rebuild_problem_from_json(data_dict)
    user_solution = ". ".join(problem.solution) + "."
    # llama_tokenizer = AutoTokenizer.from_pretrained("minglyang/llama-1b-instruct-gsmdi-mix-full")
    # print(f"problem.problem: {}")
    # llama_tokenizer.pad_token = llama_tokenizer.eos_token  # Ensure pad token is set
    tokenized_problem = tokenizer.encode(". ".join(problem.problem))
    print(tokenized_problem)
    tokenized_problem[0] = 383
    print(tokenized_problem)
    # if tokenized_problem[-1] != 128009:
        # tokenized_problem += [128009]
    # print(f"post-processing: {tokenized_problem}")
    correct, my_print, parser = true_correct(user_solution, problem)
    # print("Problem text:", problem.problem)
    # print("\nSolution text:", problem.solution)
    # print("\nCorrect?", correct)
    # if not correct:
        # print("Mistakes found:")
    # my_print.display()
    

if __name__ == "__main__":
    example_demo()
    run_validation()
