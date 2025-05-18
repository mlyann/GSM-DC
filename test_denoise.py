import os
import re
import json
import torch
import threading
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed
from tools.irr_tools_test import true_correct
from math_gen.problem_gen import Problem, Expression
from data_gen.prototype.id_gen import IdGen_PT
from typing import List
from prm_tree import tree_search
import sys
import networkx as nx
from format.format import format_prompt
import numpy as np
from datasets import load_dataset
from openai import OpenAI

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
fix_seed(42)

OP_VALUES = (2, 3)
PRM_MODEL_NAME = "PRM_MODEL"
prm_model = AutoModelForCausalLM.from_pretrained(PRM_MODEL_NAME).to(device)
print(f"PRM: {PRM_MODEL_NAME}")
MODEL_PATH = "MODEL_PATH"
model_name = MODEL_PATH.split("/")[-1]
print(f"MODEL: {MODEL_PATH}")
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
print("Loading models...")
# models = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
models = {
    op: AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    ).to(device).eval()
    for op in OP_VALUES
}
# models.eval()
print("Models loaded successfully!")
print("Loading tokenizer...")
model_tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_NAME")
model_tokenizer.pad_token = model_tokenizer.eos_token  # Ensure pad token is set
dataset = load_dataset("DATASET_PATH", data_files="DATASET_NAME.json")["train"]
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# 1) Convert a Problem object to JSON 
#    (including topological_order if it exists, 
#     and also storing whole_template).
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def expression_to_relationship(expr: Expression) -> dict:
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

    node_data = data_dict["node_data"]
    layer_counts = {}
    for key, value in node_data.items():
        node_str = value["node"]  # 例如 "(0, 0)"
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
        label = key
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

    topo_list = data_dict.get("topological_order", [])
    if topo_list:
        problem.topological_order = [
            parse_param_str(item["param"]) if isinstance(item, dict) else parse_param_str(item)
            for item in topo_list
        ]
    else:
        problem.topological_order = []

    # *** NEW CODE: Recompute n_op from topological_order ***
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


def extract_final_answer(text):
    # First try to match the expected pattern; if not, fall back to any digit sequence.
    pattern = r"<<\s*(\d+)\s*>>"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    # fallback: extract first sequence of digits
    match = re.search(r'(\d+)', text)
    if match:
        return match.group(1).strip()
    return None

def generate_problem(op=3):
    id_gen = IdGen_PT(
        style="light",
        op_style="light",
        op=op,
        perm_level=5,
        detail_level=0
    )
    id_gen.gen_prob([i for i in range(5)], p_format="pq")
    return id_gen

def generate_response(op, problem, nshots, model, cur_id_gen):
    input_text = format_prompt(True, problem, op=op, nshots=nshots, cur_id_gen=cur_id_gen, tokenizer=tokenizer, generate_problem=generate_problem)
    inputs = model_tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_len = inputs.input_ids.shape[1]
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)
        # outputs = tree_search(prm_model, model, model_tokenizer, inputs["input_ids"], 16, 4, 6*op, 0.8, device)

    generated_text = model_tokenizer.decode(outputs[0][input_len - 1:], skip_special_tokens=True)
    return generated_text.strip()

def evaluate(op: int, model, results_dict):
    results = []
    eval_dir = "eval"
    os.makedirs(eval_dir, exist_ok=True)
    output_file = f"{eval_dir}/{model_name}_op{op}_test.json"
    start = (op - 2)*300
    end = (op - 1)*300
    
    data = dataset[start: end]

    conditions = {
        "Condition_1": {"correct": 0, "incorrect": 0, "irr_correct": 0, "irr_incorrect": 0, "extracted_correct": 0, "count": 0},
        "Condition_2": {"correct": 0, "incorrect": 0, "irr_correct": 0, "irr_incorrect": 0, "extracted_correct": 0, "count": 0},
        "Condition_3": {"correct": 0, "incorrect": 0, "irr_correct": 0, "irr_incorrect": 0, "extracted_correct": 0, "count": 0}
    }

    num_problems = (end - start) // len(conditions)

    progress_bar = tqdm(
        total= num_problems*3,
        desc=f"Evaluating OP={op}",
        unit="problem"
    )

    for j in range(len(conditions.keys())):
        condition = list(conditions.keys())[j]
        for i in range(num_problems):
            problem = rebuild_problem_from_json(data[(j*100)+i])
            tokenized_problem = tokenizer.encode(". ".join(problem.problem))
            tokenized_problem[0] = 383
            # Use ans_token as ground truth answer (may include extra spaces)
            problem_text = tokenizer.decode(tokenized_problem)

            # pass into gpt tokenizer and call openai API to denoise the problem or maybe dont need
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Math Problem Simplifier. Your job is to remove any sentences that are not necessary to solve the problem. "
                        "Keep only the parts needed to compute the answer correctly."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "The number of each Arts Campus's T&T Supermarket equals 3. "
                        "The number of each Engineering Camp-us's T&T Supermarket equals 4. "
                        "The number of each Science Park's Zion Market equals 1 more than each Arts Campus's T&T Supermarket. "
                        "The number of ea-ch Arts Campus's Seafood Supermarket equals the sum of each Science Park's Zion Market, Arts Campus's T&T Supermarket and each Arts Campus's Meat Market. "
                        "The number of each Arts Campus's Meat Market equals 4 times as much as each Science Park's Zion Market. "
                        "How many Meat Market does Arts Campus have?"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "The number of each Arts Campus's T&T Supermarket equals 3. "
                        "The number of each Science Park's Zion Market equals 1 more than each Arts Campus's T&T Supermarket. "
                        "The number of each Arts Campus's Meat Market equals 4 times as much as each Science Park's Zion Market. "
                        "How many Meat Market does Arts Campus have?"
                    )
                },
                {
                    "role": "user",
                    "content": problem_text
                }]
            )
            print(completion.choices[0].message.content)
            denoised_problem_text = tokenizer.encode(completion.choices[0].message.content)
            # same as before

            predicted_solution = generate_response(op, denoised_problem_text, 5, model, None)
            irr_correct, correct, my_print, _ = true_correct(predicted_solution, problem)

            pred_final = extract_final_answer(predicted_solution)
            actual_final = int(data[(j*100)+i]["problem_info"]["final_answer"])
            extracted_correct = int(pred_final is not None and actual_final is not None and pred_final == actual_final)

            conditions[condition]["correct"] += int(correct)
            conditions[condition]["incorrect"] += int(not correct)
            conditions[condition]["irr_correct"] += int(irr_correct)
            conditions[condition]["irr_incorrect"] += int(not irr_correct)
            conditions[condition]["extracted_correct"] += extracted_correct

            results.append({
                'problem': problem_text,
                'predicted_solution': predicted_solution,
                'correct': correct,
                'irr_correct': irr_correct,
                'extracted_correct': extracted_correct,
                'input_prompt': format_prompt(True, problem_text, op=op, nshots=5, cur_id_gen=None, tokenizer=tokenizer, generate_problem=generate_problem)
            })
            progress_bar.update(1)

    progress_bar.close()

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)
    print(f"✅ OP={op} results saved to {output_file}")

# multi GPUs with multi threads
results_dict = {}
threads = []

#parallelize the branches
# for op in range(16, 23):
#     evaluate(op, models, results_dict)

for op in OP_VALUES:
    thread = threading.Thread(target=evaluate, args=(op, models[op], results_dict))
    threads.append(thread)
    thread.start()
    print(f"Thread for OP={op} started!")

for thread in threads:
    thread.join()