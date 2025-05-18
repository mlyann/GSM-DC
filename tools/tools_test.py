from math_gen.problem_gen import Problem
from tools.sol_parser import Parser
from transformers import GPT2Config
from tools.tools import subgraph_with_paths_to_node, tokenizer, MyPrint
import torch, random, copy
from typing import Union, List, Tuple, TypeVar
from data_gen.prototype.id_gen import IdGen_PT
from const.params import test_bin
import numpy as np
from const.params import mod

TARGET = TypeVar("TARGET")

def output_split(output: list, problem: Problem=None, skip_222=True):
    my_print = MyPrint()
    if problem: prob_text = " " + ". ".join(problem.problem) + "."
    my_print("\n\n\n")

    pre = 0 # the index right after 222. If there is no 222, pre = 0
    if skip_222:
        if 222 not in output:
            my_print("Warning: 222 not in the output. Start from position 0.")
        else:
            for i in output:
                pre += 1
                if i == 222:
                    break
            
            if pre == len(output):
                my_print("Error: The first 222 is the last token.")
                my_print(f"Problem: {dash*32}\n{prob_text}")
                whole_txt = tokenizer.decode(output)
                my_print(f"GPT Output: {dash*32}\n{whole_txt}")
                return my_print, whole_txt, [None, None, None]
        
    output = output[pre:]
    whole_txt = tokenizer.decode(output, skip_special_tokens=True)
    dash = "-"

    if 223 not in output or 224 not in output:
        my_print(f"Error: 223 ({223 in output}), 224 ({224 in output}).")
        my_print(f"GPT Output:", tokenizer.decode(output, skip_special_tokens=True))
        return my_print, whole_txt, [None, None, None]
    
    index_223 = output.index(223)
    index_224 = output.index(224)
    # split the list into three parts
    prob_token = output[:index_223]
    sol_token = output[index_223+1:index_224]
    ans_token = output[index_224+1:]

    prob_text = tokenizer.decode(prob_token, skip_special_tokens=True)
    sol_text = tokenizer.decode(sol_token, skip_special_tokens=True)
    ans_text = tokenizer.decode(ans_token, skip_special_tokens=True)

    if problem:
        my_print(f"Problem: {dash*32}\n{prob_text}")
        my_print(f"GPT Solution: {dash*32}\n{sol_text}")
        my_print(f"Correct Solution: {dash*32}\n{' ' + '. '.join(problem.solution) + '.'}")
        notation = "=" if ans_text == " " + str(problem.ans) else "!="
        my_print(f"Answer: {dash*32}\nGPT ans{ans_text} {notation} Correct ans {problem.ans}")
    else:
        my_print(f"Problem: {dash*32}\n{prob_text}")
        my_print(f"GPT Solution: {dash*32}\n{sol_text}")
        my_print(f"GPT Answer: {dash*32}\n{ans_text}")
    return my_print, whole_txt, [prob_text, sol_text, ans_text]

def true_correct(output, problem: Problem=None):
    correct = True
    if isinstance(output, list):
        my_print, whole_txt, decoded_txt_lst = output_split(output=output, problem=problem, skip_222=True)
        my_print()
        gpt_ans = decoded_txt_lst[2]
        if " " + str(problem.ans) != gpt_ans:
            my_print(f"Wrong final answer. gpt_ans ={gpt_ans}, problem.ans = {problem.ans}")
            correct = False
        gpt_sol = decoded_txt_lst[1]
        if gpt_sol == None or len(gpt_sol) == 0:
            my_print(f"No GPT solution")
            correct = False
            return correct, my_print, None
    elif isinstance(output, str):
        gpt_sol = output
        my_print = MyPrint()
    else:
        raise ValueError(f"output should be either list or str type.")
    
    parser = Parser(gpt_sol=gpt_sol)
    if not parser.parsed:
        my_print(gpt_sol)
        my_print(f"Could fail by the following reasons")
        if parser.duplicated_symbol:
            my_print(f"Duplicated symbol: {parser._duplicated_symbol}")
        if parser.unknown_symbol:
            my_print(f"Undefined symbol: {parser._unknown_symbol}")
        if parser.hint_cal_not_match:
            my_print(f"Hint ({parser._hint_cal_not_match[0]}) does not match {parser._hint_cal_not_match[1]}")
        if parser.illegal_def_part:
            my_print(f"Illegal def part {parser._illegal_def_part}")
        if not parser.sentence_lst:
            my_print(f"Not a single valid sentence is found")
        
        correct = False
        return correct, my_print, None
    parser.parse(problem=problem)

    gpt_ans = parser.sentence_lst[-1].ans_part.a
    correct_ans = problem.ans
    if gpt_ans != correct_ans:
        my_print(f"Wrong solution answer. gpt_ans = {gpt_ans}, problem.ans = {correct_ans}")
        correct = False
    
    wrong_ref, my_print = parser.correct_refer(my_print=my_print)
    if wrong_ref > 0:
        correct = False
    
    wrong_cal, my_print = parser.correct_cal(my_print=my_print)
    if wrong_cal > 0:
        correct = False
    
    re_define, wrong_order, my_print = parser.correct_order(my_print=my_print)
    if re_define > 0 or wrong_order > 0:
        correct = False
    # if re_define > 0 and not parser.early_stop_param:
        # raise ValueError(f"re_define ({re_define}) > 0 but not parser.early_stop_param")
    if parser.non_appear_lst:
        my_print(f"none appear lst: {parser.non_appear_lst[0]}")
        correct = False
    
    if parser.incorrect_lst:
        param, missing_but_required_param_lst, existing_but_not_required_param_lst = parser.incorrect_lst[0]
        my_print(f"Mistake happended on param {problem.get_ntn(param=param)}")
        my_print(f"missing_but_required_params: {'; '.join([problem.get_ntn(param=param_) for param_ in missing_but_required_param_lst])}")
        my_print(f"existing_but_not_required_params: {'; '.join([problem.get_ntn(param=param_) for param_ in existing_but_not_required_param_lst])}")
        correct = False
    
    if correct:
        if parser.sentence_lst[-1].param_part != problem.get_ntn(problem.ques_idx):
            my_print(f"answered param ({parser.sentence_lst[-1].param_part}) != question param ({problem.get_ntn(problem.ques_idx)})")
            correct = False
            parser.wrong_ans_param = True
        
        elif parser.sol_op < problem.n_op:
            correct = False
            my_print.display()
            print(f"parser.sol_op{parser.sol_op} < problem.n_op{problem.n_op}")
            for sol_step in parser.sol_steps:
                print(sol_step)

            raise ValueError
    
    return correct, my_print, parser

def re_ask(target: TARGET, param=None, ava_hash=test_bin, p_format="pq") -> TARGET:
    if isinstance(target, IdGen_PT):
        return_type = "id_gen"
        problem = target.problem
        id_gen = target
    elif isinstance(target, Problem):
        return_type = "problem"
        problem = target
    else:
        raise NotImplementedError
    
    if not param:
        param = random.choice(problem.all_param)
    
    # ask the param on the problem
    from tools.tools import random_topological_sort
    problem.ques_idx = param
    l, i, j, k = param
    if l == 0:
        ques = f"How many {problem.N[i+1][k]} does {problem.N[i][j]} have?"
    elif l == 1:
        ques = f"How many {problem.ln[k]} does {problem.N[i][j]} have?"
    problem.problem[-1] = ques
    original_template = problem.template
    original_problem_order = problem.problem_order
    original_problem = problem.problem
    problem.template = problem.whole_template
    problem.problem_order = random_topological_sort(problem.template)
    for param_ in problem.problem_order:
        problem.parse(param=param_, inter_only=True)
    sol_template = subgraph_with_paths_to_node(problem.template, param)
    problem.topological_order = random_topological_sort(sol_template)
    if (-1, 0, 0, 0) in problem.topological_order:
        problem.topological_order.remove((-1, 0, 0, 0))
    problem.solution = []
    problem.name_dict = {}
    problem.symbols = copy.copy(problem.all_symbols)
    for param_ in problem.topological_order:
        problem.decode(param_)
    problem.ans = problem.lookup[problem.ques_idx].a

    problem.template = original_template
    problem.problem_order = original_problem_order
    problem.problem = original_problem

    if return_type == "problem":
        problem.n_op = 0
        for param in problem.topological_order:
            num_pre = len(list(problem.template.predecessors(param)))
            problem.n_op += 1 if num_pre <= 2 else num_pre - 1
        return problem
    
    # id_gen.op_ = len(problem.topological_order)
    id_gen.op_ = 0
    for param in problem.topological_order:
        num_pre = len(list(problem.template.predecessors(param)))
        id_gen.op_ += 1 if num_pre <= 2 else num_pre - 1
    id_gen.gen_prob(ava_hash=ava_hash, p_format=p_format, problem=problem)
    return id_gen


def lora_label_with_parser(problem: Problem, parser: Parser, keys: List[str], true_correct_=False, id_gen=None): # we do not need dep_label_with_parser since the dep labels do not change with the sol
    # TODO: use parser instead of problem
    if len(keys) == 0:
        if not id_gen:
            return None, [], False
        else:
            if hasattr(id_gen, 'dep_args') or hasattr(id_gen, 'neighbor_args'):
                return None, [], True
            return None, [], False
    
    if true_correct_:
        first_wrng_param = None
    elif parser.incorrect_lst:
        first_wrng_param = parser.incorrect_lst[0][0]
        # print(f"first_wrng_param: {first_wrng_param}")
    else: # other errors
        return None, [], False
    iter_list = [] # iterable list of all correct params
    for param_name in parser.param_name_lst:
        if first_wrng_param == None or param_name != problem.get_ntn(first_wrng_param):
            iter_list.append(problem.name2param(param_name))
        else:
            break
    labels = np.zeros((1+len(iter_list), len(problem.all_param), len(keys)), dtype=int)

    in_degree = dict(problem.whole_template.in_degree())

    zero_in_degree = [v for v, d in in_degree.items() if d == 0]
    for i_, param in enumerate([(-1, 0, 0, 0)] + iter_list):
        if param not in zero_in_degree:
            return None, [], False # TODO: please refer to gdoc to debug
        zero_in_degree.remove(param)
        for v in problem.whole_template.successors(param):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                zero_in_degree.append(v)
        for j, q_param in enumerate(problem.all_param):
            for k, key in enumerate(keys):
                if key == "known":
                    if i_ > 0 and q_param in iter_list[:i_]:
                        labels[i_, j, k] = 1
                if key == "can_next":
                    if q_param in zero_in_degree:
                        labels[i_, j, k] = 1
                if key == "nece_next":
                    if q_param in zero_in_degree and q_param in problem.topological_order:
                        labels[i_, j, k] = 1
                if key == "nece":
                    if q_param in problem.topological_order:
                        labels[i_, j, k] = 1
                if key == "val":
                    if i_ > 0 and q_param in iter_list[:i_]:
                        try:
                            labels[i_, j, k] = problem.lookup[q_param].a # TODO: fix later
                        except:
                            labels[i_, j, k] = mod
                    else:
                        labels[i_, j, k] = mod
    
    return labels, iter_list, True






