from math_gen.problem_gen import Problem, Num, mod
from collections import defaultdict as dd
from tools.tools import MyPrint, idle_func
from typing import List, Dict, Set

num_world = [str(i) for i in range(mod)]

def is_num(name: str):
    if name in num_world:
        return True
    return False


class Sentence(object):
    def __init__(self, sentence="", def_part=None, param_part=None, hint_part=[], parent_part=[], sign=None, cal_part: List[Num]=[], ans_part=Num(0), idx=None) -> None:
        self.sentence = sentence
        self.def_part = def_part
        self.param_part = param_part
        self.hint_part = hint_part
        self.parent_part = parent_part
        self.sign = sign
        self.cal_part = cal_part
        self.ans_part = ans_part
        self.idx = idx
    
    def display(self):
        print(f"\n\ninfo of sentence {self.idx}")
        print(f"ntn: {self.def_part}")
        print(f"param: {self.param_part}")
        print(f"hint: {self.hint_part}")
        print(f"parents: {self.parent_part}")
        print(f"sign: {self.sign}")
        print(f"calcu: {[num.a for num in self.cal_part]}")
        print(f"ans: {self.ans_part}")
        print("\n")


class Parser(object):
    def __init__(self, gpt_sol: str, detail: str="000", if_print=False) -> None:
        self.wrong_ans_param = False
        self.retry_count = 0
        self.def_count = 0
        self.real_def_count = 0
        self.non_appear_lst = []
        self.incorrect_lst = []
        if gpt_sol[0] == " ":
            gpt_sol = gpt_sol[1:]
        if gpt_sol[-1] == ".":
            gpt_sol = gpt_sol[:-1]
        self.gpt_sol = gpt_sol
        self.sol_steps = []
        gpt_steps = gpt_sol.split(". ")
        for gpt_step in gpt_steps:
            if gpt_step.startswith("Define"):
                self.def_count += 1
            if ' BACK' in gpt_step:
                self.retry_count += 1
                continue
            if gpt_step.startswith("Define"):
                self.real_def_count += 1
            gpt_small_steps = gpt_step.split("; ")
            num_steps = len(gpt_small_steps)
            for i in range(1, num_steps-1):
                self.sol_steps.append(gpt_small_steps[i])
            last_step = gpt_small_steps[0] + gpt_small_steps[-1][4:]
            self.sol_steps.append(last_step)

        # for i in range(len(self.sol_steps)):
        #     # self.sol_steps[i] = self.sol_steps[i][1:-1]
        #     print(self.sol_steps[i])

        # self.sol_steps = gpt_sol.split(". ")
        self.sol_op = len(self.sol_steps)
        self.def_rule = detail[0]
        self.hint_rule = detail[1]
        self.cal_rule = detail[2]
        self.if_print = if_print
        self.sentence_lst: List[Sentence] = []
        self.param_dict: Dict[str, Sentence] = {}
        self.symbol_dict: Dict[str, Sentence] = {} # symbol
        # self.param_dep_map: Dict[str, Set[str]] = {}
        self.symbol_dep_map: Dict[str, Set[str]] = {}
        self.early_stop_param = None
        self.parsed = True

        self.duplicated_symbol = False # 同一个symbol用了两次
        self.unknown_symbol = False # 引用了没有出现过的symbol
        self.hint_cal_not_match = False
        self.illegal_def_part = False


        # parse each sentence
        for i, sol_step in enumerate(self.sol_steps):
            try:
                if self.if_print:
                    print(f"parse: {sol_step}")
                self.parse_sentence(sol_step=sol_step, idx=i)
                # self.sentence_lst[-1].display()
            except (NotImplementedError, IndexError, ValueError) as e:
                if self.if_print:
                    print(f"break in parser sol step:")
                    print(sol_step)
                self.parsed = False
                return
        
        # find the dependence param
        for symbol in self.symbol_dict.keys():
            self.symbol_dep_map[symbol] = self.find_dep_set(param=symbol)
        
        if not self.sentence_lst:
            if self.if_print:
                print(f"not a single valid sentence is found")
                print(sol_step)
            self.parsed = False
            return


    def parse_sentence(self, sol_step:str, idx=None):
        def_part = ""
        param_part = ""
        hint_part = []
        parent_part = []
        sign = None
        cal_part = []
        ans_part = Num(0)

        part_lst = sol_step.split(" = ")

        part = part_lst.pop(0)
        got_ntn = False
        while True:
            # print(f"part = {part}")
            # got notation (def part)
            if not got_ntn:
                if part.startswith("Define"):
                    part = part[7:]
                    part_ = part.split(" as ")
                    if len(part_) == 1:
                        def_part = part_[0]
                    else:
                        def_part = part_[1]
                        param_part = part_[0]
                else:
                    def_part = part
                
                if len(def_part) != 1:
                    self.illegal_def_part = True
                    self._illegal_def_part = def_part
                    if self.if_print:
                        print(f"Illegal def part {self._illegal_def_part}")
                    raise NotImplementedError
                got_ntn = True
                part = part_lst.pop(0)
                continue

            # exp: test if it is hint part or cal part or ans part
            skip = False
            for op_sign, op_name in zip([" + ", " - ", " * "], ["add", "sub", "mul"]):
                if op_sign in part:
                    part_ = part.split(op_sign)
                    
                    if is_num(part_[0]) and is_num(part_[1]):
                        # if it is in cal part
                        for num in part_:
                            cal_part.append(Num(num))
                        pass
                    else:
                        # if it is in hint part
                        for param in part_:
                            if not is_num(param):
                                hint_part.append(param)
                                parent_part.append(param)
                            else:
                                hint_part.append(param)
                    
                    sign = op_name
                    skip = True
                    break
            if skip:
                part = part_lst.pop(0)
                continue
            
            # not exp: test if it is hint part
            if not is_num(part):
                hint_part.append(part)
                parent_part.append(part)
                part = part_lst.pop(0)
                continue

            # ans
            if len(hint_part) == 2 and len(cal_part) == 2:
                pass
            elif len(hint_part) < 2 and len(cal_part) < 2:
                pass
            else:
                self.hint_cal_not_match = True
                self._hint_cal_not_match = (hint_part, cal_part)
                if self.if_print:
                    print(f"Hint ({self._hint_cal_not_match[0]}) does not match {self._hint_cal_not_match[1]}")
                raise NotImplementedError
            # if len(hint_part) != len(cal_part):
            #     print("hint_part and cal_part not match")
            #     print(hint_part, cal_part)
            #     raise

            ans_part = Num(part)
            sentence = Sentence(
                sentence=sol_step,
                def_part=def_part,
                param_part=param_part,
                hint_part=hint_part,
                parent_part=parent_part,
                sign=sign,
                cal_part=cal_part,
                ans_part=ans_part,
                idx=idx,
            )

            if idx is not None:
                self.sentence_lst.append(sentence)
                for parent in parent_part:
                    if parent not in self.symbol_dict:
                        self.unknown_symbol = True
                        self._unknown_symbol = parent
                        if self.if_print:
                            print(f"Undefined symbol: {self._unknown_symbol}")
                        raise NotImplementedError
                if def_part in self.symbol_dict:
                    self.duplicated_symbol = True
                    self._duplicated_symbol = def_part
                    self.parsed = False
                    if self.if_print:
                        print(f"Duplicated symbol: {self._duplicated_symbol}")
                    raise NotImplementedError
                self.symbol_dict[def_part] = sentence
                if param_part and param_part not in self.param_dict:
                    self.param_dict[param_part] = sentence
            return

    def find_dep_set(self, param: str) -> set:
        '''
        The input can be param or ntn.
        return is a set of params
        '''
        dep_set = set()
        if len(param) == 1:
            sentence = self.symbol_dict[param]
        else:
            sentence = self.param_dict[param]
        
        ntn_list = [ntn for ntn in sentence.parent_part]
        history_list = []

        while ntn_list:
            ntn = ntn_list.pop()
            if ntn not in history_list:
                history_list.append(ntn)
            else:
                continue
            sentence_ = self.symbol_dict[ntn]
            if sentence_.param_part != "":
                dep_set.add(sentence_.def_part)
                continue
            else:
                for ntn_ in sentence_.parent_part:
                    if ntn_ not in ntn_list:
                        ntn_list.append(ntn_)
        
        return dep_set

    def correct_refer(self, my_print: MyPrint=idle_func):
        wrong_refer = 0
        self.lookup = dd(set)
        for sentence in self.sentence_lst:
            if len(sentence.hint_part) == 1:
                num = sentence.ans_part.a
                if num not in self.lookup[sentence.hint_part[0]]:
                    my_print(f"{sentence.hint_part[0]} = {num} not in {self.lookup[sentence.hint_part[0]]}")
                    wrong_refer += 1
                    self.lookup[sentence.hint_part[0]].add(num)
            else:
                for i, parent in enumerate(sentence.hint_part):
                    num = sentence.cal_part[i].a
                    if not is_num(parent) and num not in self.lookup[parent]:
                        my_print(f"{parent} = {num} not in {self.lookup[parent]}")
                        wrong_refer += 1
                        self.lookup[parent].add(num)
            self.lookup[sentence.def_part].add(sentence.ans_part.a)
        return wrong_refer, my_print

    def correct_cal(self, my_print: MyPrint=idle_func):
        '''
        return the num of incorrect calculations
        '''
        wrong_cal = 0
        for sentence in self.sentence_lst:
            if sentence.sign == "add":
                if sentence.cal_part[0] + sentence.cal_part[1] != sentence.ans_part:
                    my_print(f"in {sentence.sentence}: {sentence.cal_part[0]} + {sentence.cal_part[1]} != {sentence.ans_part}")
                    wrong_cal += 1
            if sentence.sign == "sub":
                if sentence.cal_part[0] - sentence.cal_part[1] != sentence.ans_part:
                    my_print(f"in {sentence.sentence}: {sentence.cal_part[0]} - {sentence.cal_part[1]} != {sentence.ans_part}")
                    wrong_cal += 1
            if sentence.sign == "mul":
                if sentence.cal_part[0] * sentence.cal_part[1] != sentence.ans_part:
                    my_print(f"in {sentence.sentence}: {sentence.cal_part[0]} * {sentence.cal_part[1]} != {sentence.ans_part}")
                    wrong_cal += 1
        return wrong_cal, my_print

    def correct_order(self, my_print: MyPrint=idle_func):
        '''
        return the number of incorrect orders
        '''
        self.sol_order = []
        re_define = 0
        wrong_order = 0
        for sentence in self.sentence_lst:
            if sentence.def_part in self.sol_order:
                my_print(f"{sentence.sentence} has already been defined.")
                re_define += 1
            else:
                self.sol_order.append(sentence.def_part)
            for parent in sentence.parent_part:
                if parent not in self.sol_order:
                    my_print(f"{parent} has not been defined.")
                    wrong_order += 1
        
        return re_define, wrong_order, my_print

    def parse(self, problem: Problem):
        self.non_appear_lst = []
        self.non_nece_lst = []
        self.incorrect_lst = [] # elements look like (param, missing_but_required_param_lst, existing_but_not_required_param_lst)
        self.param_name_lst = []

        for sentence in self.sentence_lst:
            if sentence.param_part != "":
                if sentence.param_part not in self.param_name_lst:
                    self.param_name_lst.append(sentence.param_part)
                    duplicate = False
                else:
                    # self.early_stop_param = sentence.param_part
                    duplicate = True
                param_name = sentence.param_part
                param = problem.name2param(param_name=param_name)
                if duplicate:
                    assert param is not None
                else:
                    if param == None:
                        self.non_appear_lst.append(param_name)
                        return
                    if param not in problem.topological_order:
                        self.non_nece_lst.append(param)
                
                pre_params = [pre_param for pre_param in problem.whole_template.predecessors(param) if pre_param != (-1, 0, 0, 0)]
                gpt_params = [problem.name2param(self.symbol_dict[pre_symbol].param_part) for pre_symbol in self.symbol_dep_map[sentence.def_part]]

                missing_but_required_param_lst = []
                existing_but_not_required_param_lst = []

                for param_ in pre_params:
                    if param_ not in gpt_params:
                        missing_but_required_param_lst.append(param_)
                for param_ in gpt_params:
                    if param_ not in pre_params:
                        existing_but_not_required_param_lst.append(param_)
                
                if missing_but_required_param_lst or existing_but_not_required_param_lst:
                    self.incorrect_lst.append((param, missing_but_required_param_lst, existing_but_not_required_param_lst))
                    return
                    # print(sentence.param_part)


