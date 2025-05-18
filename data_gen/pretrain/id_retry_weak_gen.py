import random
from math_gen.problem_gen import Problem
from tools.tools import tokenizer
from const.params import retry_key_word
from data_gen.prototype.id_gen import IdGen_PT

retry_key_word_token = tokenizer.encode(" " + retry_key_word + ".", return_tensors='pt')[0].tolist()

class IdGen(IdGen_PT):
    def __init__(self, max_op=10, max_edge=15, op=None, perm_level: str = None, detail_level: str = None, retry_rate: int = 0.02, self_contain=True) -> None:
        super().__init__('light', 'light', max_op, max_edge, op, perm_level, detail_level)
        self.retry_rate = retry_rate
        self.self_contain = self_contain
    
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
        # self.labels = self.problem.lora_label(keys=["can_next"])
        
        self.new_sols = []

        sol_indices = [self.problem.all_param.index(param) for param in self.problem.topological_order]
        
        for i, param_sol in enumerate(self.sols):
            if self.self_contain:
                indices = sol_indices[i:]
            else:
                indices = sol_indices[i+1:]
            while indices:
                if random.random() < self.retry_rate:
                    wrng_param_index = random.choice(indices)
                    indices.remove(wrng_param_index)
                    # non_appear_list.remove(wrng_param_index)
                    self.new_sols.append([2896, 500] + self.param_tokens[wrng_param_index] + [355] + retry_key_word_token)
                else:
                    break
            
            self.new_sols.append(tokenizer.encode(param_sol, return_tensors='pt')[0].tolist())
        
        self.sol_token = sum(self.new_sols, start=[])

        self.token_id = [222] + self.prob_token + [223] + self.sol_token + [224] + self.ans_token + [50256]
        self.prob_id = [50256] + [222] + self.prob_token + [223]
