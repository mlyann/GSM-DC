import random
import numpy as np
from math_gen.problem_gen import Problem
from tools.tools import tokenizer
from const.params import retry_key_word
from typing import List
from torch.utils.data import Dataset
from data_gen.prototype.id_gen import IdGen_PT

retry_key_word_token = tokenizer.encode(" " + retry_key_word + ".", return_tensors='pt')[0].tolist()

class IdGen(IdGen_PT):
    def __init__(self, max_op=10, max_edge=15, op=None, perm_level: str = None, detail_level: str = None, retry_rate: int = 0.02) -> None:
        super().__init__('light', 'light', max_op, max_edge, op, perm_level, detail_level)
        self.retry_rate = retry_rate
    
    def gen_prob(self, ava_hash, p_format: str, problem: Problem=None):
        super().gen_prob(ava_hash, p_format, problem=problem)
    
    def insert_retry(self):
        self.sols = []
        sol_str = ""
        for sol in self.problem.solution:
            sol_str += " " + sol + "."
            if sol.startswith("Define"):
                self.sols.append(sol_str)
                sol_str = ""
        
        if len(self.sols) != len(self.problem.solution):
            raise ValueError(f"len(self.sols) != len(self.problem.solution):\n{len(self.sols)} != {len(self.problem.solution)}")
        
        self.param_tokens = [tokenizer.encode(" " + self.problem.get_ntn(param), return_tensors='pt')[0].tolist() for param in self.problem.all_param]
        num_params = len(self.param_tokens)
        non_appear_list = list(range(num_params))
        self.labels = self.problem.lora_label(keys=["can_next"])
        
        self.new_sols = []
        
        for i, param_sol in enumerate(self.sols):
            indices:List[int] = np.where(self.labels[i, :, 0] == 0)[0].tolist()
            indices = [idx for idx in indices if idx in non_appear_list]
            while indices:
                if random.random() < self.retry_rate:
                    wrng_param_index = random.choice(indices)
                    indices.remove(wrng_param_index)
                    # non_appear_list.remove(wrng_param_index)
                    self.new_sols.append([2896, 500] + self.param_tokens[wrng_param_index] + [355] + retry_key_word_token)
                else:
                    break
            
            self.new_sols.append(tokenizer.encode(param_sol, return_tensors='pt')[0].tolist())
            idx = self.problem.all_param.index(self.problem.topological_order[i])
            non_appear_list.remove(idx)
        
        self.sol_token = sum(self.new_sols, start=[])

        self.token_id = [222] + self.prob_token + [223] + self.sol_token + [224] + self.ans_token + [50256]
        self.prob_id = [50256] + [222] + self.prob_token + [223]

