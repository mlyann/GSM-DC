from matplotlib.patches import ArrowStyle
from typing import Dict, Callable, Any, Tuple
import numpy as np
import random, copy
import networkx as nx
from tools.tools import random_topological_sort
from matplotlib.lines import Line2D

class Graph():
    def __init__(self, d, w0, w1, e, p, perm=True, dist: Dict[str, Callable[[], Any]]=None) -> None:
        '''
        d is the generator for depth
        w0, w1 is the min and max width
        e is the generator for the number of edges
        p is the probability of adding vertex when it's possible
        if e does not satisfy: (d-1) * w1 ** 2 >= e >= (d-1) * w0,
        choose the feasible e closest to the original e.
        '''
        if (d-1) * w1 ** 2 < e:
            e = (d-1) * w1 ** 2 
        elif e < (d-1) * w0:
            e = (d-1) * w0

        self.d = d
        self.w0 = w0
        self.w1 = w1
        self.e = min(e + 1, (self.d-1)*self.w1**2)
        self.p = p
        self.perm = perm
        self.record = {}
        self.rand = (-1, 0, 0, 0)
        self.dist = dist if dist is not None else {}

    def init(self):
        self.topological_order = None
        self.problem_order = None
        # construct self.l: layer list
        self.l = np.ones(self.d, dtype=int) * self.w0

        while True:
            if self.valid == "add vertex":
                _ = self.add_vertex()
            elif self.valid == "stop":
                break
            elif random.random() < self.p:
                is_full = self.add_vertex()
                if is_full:
                    break
            else:
                break

        # TODO: select some nodes to be unique
        self.unique = []

        eoc = [(i, j, k) for i in range(self.d-1) for j in range(self.l[i]) for k in range(self.l[i+1])] # edges of the complement graph
        self.record['remain'] = [] # add elements later
        self.record['inter'] = [(i, j, k) for i in range(self.d-1) for j in range(self.l[i]) for k in range(i+1, self.d)]
        self.record['chosen'] = [] # 4-tuple

        # construct self.G: graph matrices list
        self.G = []
        n_exist = 0
        for i in range(self.d - 1):
            self.G.append(np.zeros((self.l[i], self.l[i+1]), dtype=bool))
            for k in range(self.l[i+1]):
                j = np.random.choice(self.l[i])
                self.G[i][j, k] = True
                eoc.remove((i, j, k))
                self.record['remain'].append((i, j, k))
            n_exist += self.l[i + 1]
        n_remain = self.e - n_exist
        eor = random.sample(eoc, n_remain) # edge of remaining
        self._extra_pool = eoc[:]
        self.record['remain'] += eor
        for i, j, k in eor:
            self.G[i][j, k] = True

        self.graph = nx.DiGraph()
        for i in range(self.d):
            for j in range(self.l[i]):
                self.graph.add_node((i, j), unique=False)
        
        for i in range(self.d-1):
            for j in range(self.l[i]):
                for k in range(self.l[i+1]):
                    if self.G[i][j, k]:
                        self.graph.add_edge((i, j), (i+1, k), chosen=False)
        
        self.total = len(self.record['remain'])
        for i in range(self.d-1):
            for j in range(self.l[i]):
                count = self.G[i][j, :].sum()
                count_multi = max(2 * count - 1, 0)
                if count > 1:
                    count -= 1
                self.total += count_multi * (self.d - i - 2) + count

    def display(self, detail=False):
        print(f"Graph level: {self.l}")
        info = f"Graph info: d={self.d}, w0={self.w0}, w1={self.w1}, e={self.e}"
        for attr in ['n_inter', 'n_op_min', 'n_op']:
            if hasattr(self, attr):
                info += f", {attr}={getattr(self, attr)}"
        for attr in ['unique']:
            if hasattr(self, attr):
                print(f"{attr}={getattr(self, attr)}")
        for attr in ['topological_order', 'problem_order']:
            if hasattr(self, attr):
                params = getattr(self, attr)
                if params is not None:
                    print(f"{attr}={params}, op={self.op_num(params)}")
                else:
                    print(f"{attr}=None")
        info += f", total={self.total}."
        print(info)
        if detail:
            print(self.record['remain'])
            print(self.record['inter'])
            print(self.record['chosen'])
    
    def add_vertex(self):
        pool = [i for i, x in enumerate(self.l) if x < self.w1]
        if pool:
            index = random.choice(pool)
            self.l[index] += 1
            return False
        else:
            return True

    @property
    def n_param(self):
        if hasattr(self, 'problem_order'):
            if self.rand in self.problem_order:
                return len(self.problem_order) - 1
            return len(self.problem_order)
        if hasattr(self, 'topological_order'):
            if self.rand in self.topological_order:
                return len(self.topological_order) - 1
            return len(self.topological_order)
        return len(self.record['chosen'])

    @property
    def valid(self):
        max_v = 0
        min_v = 0
        for i in range(self.d - 1):
            max_v += self.l[i] * self.l[i+1]
            min_v += self.l[i+1]
        if max_v < self.e:
            return "add vertex"
        elif min_v == self.e:
            return "stop"
        else:
            return "ok"
    
    def try_inter(self, inter_p, start=True):
        i, j, k = inter_p
        if start:
            self.temp_record = copy.deepcopy(self.record)
        # print("remove: ", inter_p, self.temp_record['inter'])
        self.temp_record['inter'].remove(inter_p)
        self.temp_record['chosen'].append((1, i, j, k))
        count = 0
        if i + 1 == k:
            n_off_spring = 0
            for x in range(self.l[i+1]):
                if self.G[i][j, x]:
                    n_off_spring += 1
                    if (i, j, x) in self.temp_record['remain']:
                        count += 1
                        self.temp_record['remain'].remove((i, j, x))
                        self.temp_record['chosen'].append((0, i, j, x))
            if n_off_spring > 1:
                n_off_spring -= 1
            else:
                n_off_spring = 1
            return count + n_off_spring
        else:
            n_off_spring = 0
            for x in range(self.l[i+1]):
                if self.G[i][j, x]:
                    n_off_spring += 1
                    if (i+1, x, k) in self.temp_record['inter']:
                        count += self.try_inter((i+1, x, k), False)
                    if (i, j, x) in self.temp_record['remain']:
                        count += 1
                        self.temp_record['remain'].remove((i, j, x))
                        self.temp_record['chosen'].append((0, i, j, x))
            return count + max(2 * n_off_spring - 1, 1)

    def choose_param(self, n, m):
        '''
        n: params related to internal parameters
        m: all params
        m >= n.
        '''
        if n > self.total:
            # print(f"warning: n({n}) > total({self.total})")
            n = self.total
        self.n_inter = n
        n_fix = m - n
        previous_record = self.record
        previous_count = 0
        lowest = 1
        highest = self.d - 1
        while n > 0:
            for diff in range(1, self.d):
                try_set = [(i, j, k) for i, j, k in self.record['inter'] if k - i == diff]
                if not try_set:
                    if diff == lowest:
                        lowest += 1
                    elif diff == highest:
                        highest -= 1
                if try_set:
                    i, j, k = random.choice(try_set)
                    count = self.try_inter((i, j, k))
                    # print(f"Try diff={diff}, lowest={lowest}, highest={highest}, (i, j, k)={i, j, k}, n={n}, count={count}.")
                    if diff == lowest and count > n:
                        self.n_inter -= n
                        n_fix_v2 = min(n + n_fix, len(self.record['remain']))
                        fix = random.sample(self.record['remain'], n_fix_v2)
                        for i, j, k in fix:
                            self.record['chosen'].append((0, i, j, k))
                            self.record['remain'].remove((i, j, k))
                        self.n_op_min = m - n - n_fix + n_fix_v2
                        return
                    if count <= n:
                        if diff < highest:
                            previous_record = self.temp_record
                            previous_count = count
                            previous_choose = (i, j, k)
                        else:
                            self.record = self.temp_record
                            # print(f"Choose {i, j, k}. n={n}, count={count}.")
                            n -= count
                            break
                    else:
                        self.record = previous_record
                        # print(f"Choose {previous_choose}. n={n}, count={previous_count}.")
                        n -= previous_count
                        break
        n_fix_v2 = min(n_fix, len(self.record['remain']))
        fix = random.sample(self.record['remain'], n_fix_v2)
        for i, j, k in fix:
            self.record['chosen'].append((0, i, j, k))
            self.record['remain'].remove((i, j, k))
        self.n_op_min = m - n_fix + n_fix_v2

    def reasonable_sort(self, first: int=-1):
        '''
        sort the used parameters and add additional edges to make all used parameters necessary
        '''
        if 'p0' in self.dist:
            p0: float = self.dist['p0']()
        else:
            p0 = random.random() # the probability of quoting existing parameters
        
        if 'p1p2' in self.dist:
            pair: Tuple[float, float] = self.dist['p1p2']()
            p1, p2 = pair
        else:
            p1 = abs(np.random.randn()) # bias on prefered parameters
            p2 = p1
        # Step 1: Compute the out-degree for each vertex
        out_degree = dict(self.template.out_degree())

        # Step 1: Add vertices with out-degree 0 to the set
        zero_out_degree = [v for v, d in out_degree.items() if d == 0]
        stack = []
        remain = list(self.template.nodes())
        if self.rand in remain:
            remain.remove(self.rand)

        # Step 2: While the set is not empty
        topological_order = []

        def choose_param(param):
            '''
            choose the parameters which will be added into the stack
            '''
            remain.remove(param)
            zero_out_degree.remove(param)
            for v in self.template.predecessors(param):
                out_degree[v] -= 1
                if out_degree[v] == 0:
                    zero_out_degree.append(v)
            if param in stack:
                stack.remove(param)
            need_to_pick = True
            for i in stack:
                if i in zero_out_degree:
                    need_to_pick = False
            if param[0] == 0 and (param[1]+1, param[3]) not in self.unique:
                # Randomly choose a param from 'remain' list. Bias on its in-degree and if it is internal parameter.
                if (random.random() < p0 or need_to_pick) and remain:
                    if need_to_pick:
                        pool_temp = zero_out_degree
                    else:
                        pool_temp = remain
                    values = np.zeros(len(pool_temp))
                    for i, ele in enumerate(pool_temp):
                        if ele[0] == 1:
                            values[i] += p1
                        if ele not in stack:
                            values[i] += p2
                    e_x = np.exp(values - np.max(values))
                    probabilities = e_x / e_x.sum()
                    random_index = np.random.choice(len(pool_temp), p=probabilities)
                    random_element = pool_temp[random_index]
                    self.template.add_edge(random_element, param)
                    # print(f"add edge from {random_element} to {param}.")
                    if random_element not in stack:
                        stack.append(random_element)

            elif param[0] == 1:
                for v in self.template.predecessors(param):
                    if v not in stack:
                        stack.append(v)
        
        start = True
        while remain:
            if start:
                if first == 1:
                    pool = [param for param in zero_out_degree if param[0] == 1]
                elif first == 0:
                    pool = [param for param in zero_out_degree if param[0] == 0]
                else:
                    pool = zero_out_degree
                start = False
            else:
                pool = [param for param in zero_out_degree if param in stack]
            '''print("--------")
            print(zero_out_degree)
            print(stack)'''
            if not pool:
                return False
            picked = random.choice(pool)
            topological_order.append(picked)
            # print(f"from {pool} pick {picked}. Out degree is {out_degree[picked]}")
            choose_param(param=picked)
        self.topological_order = topological_order
        self.topological_order.reverse()
        return True

    def design(self, s, max_param=4):
        '''
        s is the number of deduction steps.
        '''
        self.n_op = max(self.n_op_min, s)
        max_extra = []
        addable = []
        self.extra = []
        for i, param in enumerate(self.topological_order):
            max_extra.append(0)
            self.extra.append(0)
            if param[0] == 0 and (param[1]+1, param[3]) not in self.unique:
                if max_param is not None:
                    max_extra[-1] += max(min(i + 1, max_param) - 2, 0) # if max_param <= 2, then no addable params.
                else:
                    max_extra[-1] = max(i - 1, 0)
                if max_extra[-1] > 0:
                    addable.append(i)
        
        self.n_op = min(self.n_op - self.n_op_min, sum(max_extra)) + self.n_op_min
        n_fix = self.n_op - self.n_op_min
        while n_fix > 0:
            index = random.choice(addable)
            n_fix -= 1
            self.extra[index] += 1
            if self.extra[index] == max_extra[index]:
                addable.remove(index)
        
        # print(max_extra)
        # print(self.extra)

        self.template.add_node(self.rand)
        self.template.nodes[self.rand]['type'] = -1
        
        for i, param in enumerate(self.topological_order):
            '''
            if max_extra[i] == 0: if it is not pointed from a previous node, you can assign it one with some probability
            if max_extra[i] > 0: randomly select max_extra[i] or max_extra[i] + 1 previous nodes. If only max_extra[i] nodes, please also add random node.
            '''
            self.template.nodes[param]['type'] = 0
            l_, i_, j_, k_ = param
            if l_ == 0:
                self.graph.edges[(i_, j_), (i_+1, k_)]['chosen'] = True
                if (i_+1, k_) in self.unique:
                    continue
                pool = [self.rand] + self.topological_order[:i]

                # determine the number of parameters
                if self.extra[i] == 0:
                    n_sample = 1 if random.random() < 0.5 else 2
                elif self.extra[i] > 0:
                    n_sample = self.extra[i] + 2
                if max_param is not None:
                    n_sample = min(n_sample, i + 1, max_param)
                else:
                    n_sample = min(n_sample, i + 1)
                # print("0", n_sample, len(pool))
                
                # choose parameters
                if n_sample != len(pool):
                    for v in self.template.predecessors(param):
                        pool.remove(v)
                        n_sample -= 1
                    # print("1", n_sample, len(pool))
                    if random.random() < 0.5 and n_sample > 0:
                        n_sample -= 1
                        self.template.add_edge(self.rand, param)
                        pool.remove(self.rand)
                    # print("2", n_sample, len(pool))
                    pool = random.sample(pool, n_sample)
                for v in pool:
                    if v not in self.template.predecessors(param):
                        self.template.add_edge(v, param)

    def design_unused(self, max_param=4, target_range=None):
        '''
        difference between self.problem_order and self.record['chosen']:
        self.problem_order only contains parameters appeared in the problem in a proper order,
        self.record['chosen'] does not contain order information and it also contains internal parameters that can be derived now.
        '''
        low, high = (None, None) if target_range is None else target_range
        self.problem_order = copy.deepcopy(self.topological_order)
        self.independent = []
        self.problem_order = copy.deepcopy(self.topological_order)
        self.independent = []
        while self.record['remain']:
            if high is not None:
                n_irrel = len(self.find_irrelevant_nodes_edges()[0])
                if n_irrel >= high:
                    break          # 已达上限，停止注入     
            i, j, k = random.choice(self.record['remain'])
            param = (0, i, j, k)
            self.template.add_node(param)
            if (i+1, k) in self.unique:
                self.record['remain'].remove((i, j, k))
                self.record['chosen'].append(param)
                self.independent.append(param)
                self.problem_order.append(param)
                self.template.nodes[param]['type'] = 2
                continue
            if 'p3' in self.dist:
                p3: float = self.dist['p3']()
            else:
                p3 = 0.5
            if random.random() < p3:
                independent = True
            else:
                independent = False
            if independent:
                pool = self.independent
            else:
                pool = self.record['chosen']
            n_sample = 1
            max_sample_ = len(pool) + 1
            if max_param:
                max_sample = min(max_sample_, max_param)
            else:
                max_sample = max_sample_
            while n_sample < max_sample:
                if random.random() > 0.5:
                    n_sample += 1
                else:
                    break
            
            self.template.nodes[param]['type'] = 2
            if n_sample == max_sample_:
                self.template.add_edge(self.rand, param)
            else:
                if random.random() < 0.5:
                    self.template.add_edge(self.rand, param)
                    n_sample -= 1
                pool = random.sample(pool, n_sample)
            for v in pool:
                self.template.add_edge(v, param)
                self.add_param(v)
                # print("before", param, v, self.template.nodes[param]['type'], self.template.nodes[v]['type'])
                if self.template.nodes[v]['type'] in [0, 1]:
                    self.template.nodes[param]['type'] = 1
                # print("after", param, v, self.template.nodes[param]['type'], self.template.nodes[v]['type'])
            self.record['remain'].remove((i, j, k))
            self.record['chosen'].append(param)
            self.fill_chosen(param) # add internal paramters that can be added now
            if independent:
                self.independent.append(param)
                self.problem_order.append(param)
            else: self.problem_order.append(param)
    def gen_debug(self, n, m, s, first=-1, max_param=4):
        '''
        n is the operation needed for internal parameters
        m is the total number of minimal required operations
        s is the total number of operations
        first determine the type of the final question
        max_param determine the maximal number of parametered can appear in a single sentence
        '''
        while True:
            self.init()
            self.assign_unique_debug()
            self.choose_param(n, m)
            self.setup_template()
            # print("finished")
            valid = self.reasonable_sort(first=first)
            if valid:
                self.design(s, max_param=max_param)
                self.fill_all()
                self.ques_pos = len(self.topological_order)
                self.ques_idx = self.topological_order[-1]
                self.design_unused(max_param=max_param)
                if self.perm:
                    self.problem_order = random_topological_sort(self.template)
                return

    def fill_all(self):
        # fill all the possible internal parameters
        for upper in range(self.d-1):
            lower = upper + 1
            for j in range(self.l[upper]):
                off_springs = [o for o in range(self.l[upper+1]) if self.G[upper][j, o]]
                out_edge = self.graph.out_edges((upper, j), data=True)
                degree_now = sum(1 for edge in out_edge if edge[2]['chosen'])
                if len(off_springs) == degree_now:
                    if (upper, j, lower) in self.record['inter']:
                        self.record['inter'].remove((upper, j, lower))
                        self.record['chosen'].append((1, upper, j, lower))
        for upper in range(self.d-1):
            stamp = False
            for lower in range(upper+2, self.d):
                for j in range(self.l[upper]):
                    if (upper, j, upper+1) not in self.record['inter']:
                        off_springs = [o for o in range(self.l[upper+1]) if self.G[upper][j, o]]
                        for off_spring in off_springs:
                            if (upper+1, off_spring, lower) in self.record['inter']:
                                stamp = True
                                break
                        if stamp:
                            break
                        if (upper, j, lower) in self.record['inter']:
                            self.record['inter'].remove((upper, j, lower))
                            self.record['chosen'].append((1, upper, j, lower))

    def fill_chosen(self, param):
        '''
        add internal parameter to chosen set based on the current chosen set.
        '''
        _, i, j, k = param
        self.graph.edges[(i, j), (i+1, k)]['chosen'] = True
        off_springs = [o for o in range(self.l[i+1]) if self.G[i][j, o]]
        degree = len(off_springs)
        out_edge = self.graph.out_edges((i, j), data=True)
        degree_now = sum(1 for edge in out_edge if edge[2]['chosen'])
        if degree == degree_now:
            self.record['inter'].remove((i, j, i+1))
            # print("remove0", param, (i, j, i+1))
            self.record['chosen'].append((1, i, j, i+1))

            stamp = False
            for lower in range(i+2, self.d):
                # print(param, lower)
                for off_spring in off_springs:
                    if (i+1, off_spring, lower) in self.record['inter']:
                        stamp = True
                        # print("break", param, (i+1, off_spring, lower))
                        break
                if stamp:
                    break
                # print('--------')
                # print(param, (i, j, lower))
                # print(self.record['inter'])
                # print(self.record['chosen'])
                self.record['inter'].remove((i, j, lower))
                # print("remove1", param, (i, j, lower))
                self.record['chosen'].append((1, i, j, lower))
                # print("finished")

            for upper in range(i-1, -1, -1):
                stamp = False
                for lower in range(i+1, self.d):
                    for j in range(self.l[upper]):
                        if (upper, j, upper+1) not in self.record['inter']:
                            off_springs = [o for o in range(self.l[upper+1]) if self.G[upper][j, o]]
                            for off_spring in off_springs:
                                if (upper+1, off_spring, lower) in self.record['inter']:
                                    stamp = True
                                    break
                            if stamp:
                                break
                            if (upper, j, lower) in self.record['inter']:
                                self.record['inter'].remove((upper, j, lower))
                                # print("remove2", param, (upper, j, lower))
                                self.record['chosen'].append((1, upper, j, lower))

    def add_param(self, param):
        '''
        check if the paramter is in the self.problem_order.
        if not, add all the edged related to this param in the template.
        '''
        l, i, j, k = param
        if l == 1 and param not in self.problem_order:
            self.template.nodes[param]['type'] = 2
            if i+1 == k:
                for off_spring in [o for o in range(self.l[i+1]) if self.G[i][j, o]]:
                    self.template.add_edge((0, i, j, off_spring), param)
                    if self.template.nodes[(0, i, j, off_spring)]['type'] in [0, 1]:
                        self.template.nodes[param]['type'] = 1
                self.problem_order.append(param)
                return
            else:
                for off_spring in [o for o in range(self.l[i+1]) if self.G[i][j, o]]:
                    self.template.add_edge((0, i, j, off_spring), param)
                    if self.template.nodes[(0, i, j, off_spring)]['type'] in [0, 1]:
                        self.template.nodes[param]['type'] = 1
                    self.template.add_edge((1, i+1, off_spring, k), param)
                    if (1, i+1, off_spring, k) not in self.problem_order:
                        self.add_param((1, i+1, off_spring, k))
                    if self.template.nodes[(1, i+1, off_spring, k)]['type'] in [0, 1]:
                        self.template.nodes[param]['type'] = 1
                self.problem_order.append(param)
                return

    def assign_unique_debug(self):
        '''
        assign uniqueness to parameters according to data.
        just for debugging
        '''
        for v in self.graph.nodes():
            if random.random() < 0.:
                self.unique.append(v)
                self.graph.nodes[v]['unique'] = True

    def op_num(self, params):
        '''
        compute the number of operations needed for these parameters
        '''
        print("params", params)
        n_op = 0
        for param in params:
            op = self.template.in_degree[param]
            if op <= 2:
                n_op += 1
            else:
                n_op += op - 1
        return n_op

    def draw_template(self, ax=None, labels=False, rotate_seed = None):
        import matplotlib.patches as mpatches
        from tools.tools import wrap_label
        # define color map:
        color = {
            -1: "#E6E6FA", # Lavender
            #0: "#FF7F50", # Coral
            0: "#40E0D0", # Coral
            1: "#DDA0DD", # Plum
            #2: "#40E0D0", # Turquoise
            2: "#DDA0DD", # Plum
        }
        def alpha(param):
            if param[0] == 1:
                return 1
            if (param[1]+1, param[3]) in self.unique:
                return 0.5
            return 1
        handles = []
        # define pos and color
        pos = {self.rand: (0, 0)}
        if hasattr(self, 'partial_template'):
            sorted_param = copy.deepcopy(self.partial_param)
            # sorted_param = list(nx.topological_sort(self.partial_template))
            if self.rand in sorted_param:
                sorted_param.remove(self.rand)
            for order, node in enumerate(sorted_param):
                pos[node] = (np.sin(2*np.pi*(order+1)/len(sorted_param)), np.cos(2*np.pi*(order+1)/len(sorted_param)))
            nx.draw_networkx_nodes(self.partial_template, pos, ax=ax)
            nx.draw_networkx_edges(self.partial_template, pos, ax=ax)
            nx.draw_networkx_labels(self.partial_template, pos, font_size=8, ax=ax)  # set font_size here
            return
        elif self.problem_order:
            #handles.append(mpatches.Patch(color="#E6E6FA", label='random paramter'))
            #handles.append(mpatches.Patch(color="#FF7F50", label='used paramter'))
            #handles.append(mpatches.Patch(color="#DDA0DD", label='related paramter'))
            #handles.append(mpatches.Patch(color="#40E0D0", label='unrelated paramter'))
            #handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color[-1], markersize=10, label='random generator'))
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color[0], markersize=10, label='neccesary paramter'))
            #handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor=color[1], markersize=10, label='related paramter'))
            handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor=color[2], markersize=12, label='unused paramter'))
            if self.perm:
                #handles.append(mpatches.Patch(color="#00BFFF", label='question paramter'))
                handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor="#00BFFF", markersize=17, label='question paramter'))
                sorted_param = copy.deepcopy(self.problem_order)
                if rotate_seed is not None:
                    random.Random(rotate_seed).shuffle(sorted_param)
                # if hasattr(self, "ori_order"): sorted_param = copy.deepcopy(self.ori_order)
                if self.rand in sorted_param:
                    sorted_param.remove(self.rand)
                for order, node in enumerate(sorted_param):
                    pos[node] = (np.sin(2*np.pi*(order)/(self.n_param)), np.cos(2*np.pi*(order)/(self.n_param)))
                node_colors = []
                for node in self.template.nodes(data=True):
                    if node[0] == self.ques_idx:
                        node_colors.append("#00BFFF") # Deep Sky Blue
                    else:
                        node_colors.append(color[node[1]['type']])
            else:
                used = self.ques_pos
                unused = len(self.problem_order) - used
                for order, node in enumerate(self.topological_order):
                    pos[node] = (np.sin(np.pi*(order+0.5)/used), np.cos(np.pi*(order+0.5)/used))
                for order, node in enumerate(self.problem_order[used:]):
                    pos[node] = (np.sin(np.pi+np.pi*(order+0.5)/unused), np.cos(np.pi+np.pi*(order+0.5)/unused))
                node_colors = [color[node[1]['type']] for node in self.template.nodes(data=True)]
            node_alphas = [alpha(node) for node in self.template.nodes()] # so^>v<dph8
        else:
            if self.topological_order:
                sorted_param = self.topological_order
            else: sorted_param = random_topological_sort(self.template)
            if self.rand in sorted_param:
                sorted_param.remove(self.rand)
            for order, node in enumerate(sorted_param):
                pos[node] = (np.sin(2*np.pi*(order+1)/self.n_param), np.cos(2*np.pi*(order+1)/self.n_param))
            node_colors = "#1f78b4"
            handles.append(mpatches.Patch(color='#1f78b4', label='problem paramter'))
            node_alphas = 1 # so^>v<dph8
        
        edge_color = []
        for edge in self.template.edges():
            # print(edge)
            if edge[1][0] == 1:
                edge_color.append("red")
            else:
                edge_color.append("black")

        #nx.draw_networkx_nodes(self.template, pos, ax=ax, node_color=node_colors, alpha=node_alphas)
        for c, shape in zip(list(color.values())+["#00BFFF"], ['o', 'o' ,'^' ,'^' ,'*']):
            cur_nodes = [node[0] for node in self.template.nodes(data=True) if color[node[1]['type']] == c and node[0] != self.ques_idx or node[0] == self.ques_idx and c=="#00BFFF"]
            #print(cur_nodes)
            nx.draw_networkx_nodes(self.template, pos, ax=ax, nodelist=cur_nodes, node_color=c, node_shape=shape, node_size=180 if c != color[-1] else 560)
        nx.draw_networkx_edges(self.template, pos, ax=ax, edge_color=edge_color, min_target_margin=15, min_source_margin=0, arrowstyle=ArrowStyle.Fancy(head_length=.5, head_width=.5, tail_width=.2))
        if labels:
            max_chars_per_line = 12
            labels = {}
            for param in self.problem_order:
                if param == (-1, 0, 0, 0):
                    labels[param] = "RNG"
                else:
                    param_name = self.get_ntn(param)
                    labels[param] = wrap_label(param_name, max_chars_per_line)
            
            nx.draw_networkx_labels(self.template, pos, labels=labels, font_size=8, font_family="sans-serif", ax=ax)  # set font_size here
        else:
            nx.draw_networkx_labels(self.template, pos, font_size=8, ax=ax)  # set font_size here

        from matplotlib.legend_handler import HandlerPatch
        # Define a custom handler for the arrow
        class HandlerArrow(HandlerPatch):
            def create_artists(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize, trans):
                # Calculate the center of the legend handle area
                x = width / 2.0
                y = height / 2.0

                # Create a FancyArrowPatch object as the legend handle
                p = mpatches.FancyArrowPatch((xdescent + x + 10, ydescent + y), (xdescent + x - 10, ydescent + y),
                                    mutation_scale=15, arrowstyle=ArrowStyle.Fancy(head_length=.5, head_width=.5, tail_width=.2), color=orig_handle.get_facecolor(),
                                    transform=trans)
                
                return [p]

        handles += [mpatches.FancyArrowPatch((0, 0), (1, 1), color='red', arrowstyle='->', label='abstract dependency')]
        handles += [mpatches.FancyArrowPatch((0, 0), (1, 1), color='black', arrowstyle='->', label='instance dependency')]
        handles = [ handles[0], handles[3], handles[1], handles[4], handles[2] ]
        handler_map = {mpatches.FancyArrowPatch: HandlerArrow()}
        # create legend:
        ax.legend(handles=handles, handler_map=handler_map, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)

    def draw_structure(self, ax=None):
        import matplotlib.patches as mpatches
        # define color map:
        color = {
            False: "#1f78b4",
            True: "#228B22" #"#FF7F50",
        }
        pos = {}
        for i in range(self.d):
            for j in range(self.l[i]):
                pos[(i, j)] = (j, self.d-1-i)
        node_colors = [color[node[1]['unique']] for node in self.graph.nodes(data=True)]
        if hasattr(self, "partial_inst_param"):
            print("Has")
            selected = set()
            for l, i, j, k in self.partial_param:
                if l == 0:
                    selected.add((i, j))
                    selected.add((i+1, k))
            map_to_color = {True: 1, False: 0.2}
            node_alphas = [map_to_color[node in selected] for node in self.graph.nodes()]
        else:
            node_alphas = 1

        nx.draw(self.graph, pos, node_color=node_colors, with_labels=True, font_weight='bold', ax=ax, alpha=node_alphas)

        # create legend:
        unique_patch = mpatches.Patch(color='#228B22', label='unique item')
        duplicate_patch = mpatches.Patch(color='#1f78b4', label='duplicatable item')
        ax.legend(handles=[unique_patch, duplicate_patch], loc='upper center', bbox_to_anchor=(0.5, -0.), ncol=2)

    def draw(self, file=None, zoom_ratio = 1.0, rotate_seed = None):
        # import matplotlib
        # matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import os
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(15*zoom_ratio, 5*zoom_ratio))
        gs = GridSpec(1, 2, width_ratios=[1, 2], height_ratios=[1], figure=fig)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        #fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        self.draw_structure(axes[0])
        axes[0].set_title("Structure Graph", fontsize=18*zoom_ratio, fontweight='bold')
        self.draw_template(axes[1], labels=True, rotate_seed = rotate_seed)
        axes[1].set_title("Dependency Graph", fontsize=18*zoom_ratio, fontweight='bold')
        # newly added on 12-21
        # plt.subplots_adjust()
        plt.tight_layout()
        if not os.path.exists('plot'):
            os.makedirs('plot')
        if file != None:
            plt.savefig(f'plot/{file}.png')
        else:
            plt.savefig('plot/my_plot.png')
        # plt.show()

    def setup_template(self):
        '''
        Given self.record['chosen'], construct the smallest disgram for all chosen parameters.
        '''
        self.template = nx.DiGraph()
        for param in self.record['chosen']:
            self.template.add_node(param)
        # self.template.add_node(self.rand)
        
        for l, i, j, k in self.record['chosen']:
            if l == 1:
                for x in range(self.l[i+1]):
                    if self.G[i][j, x]:
                        self.template.add_edge((0, i, j, x), (1, i, j, k))
                if k - i > 1:
                    for x in range(self.l[i+1]):
                        if self.G[i][j, x]:
                            self.template.add_edge((1, i+1, x, k), (1, i, j, k))



def auto_easy(max_op=5):
    '''
    auto gen easy problem template
    '''
    d = random.choice((2, 3))
    t0 = random.choice((2, 3))
    t1 = random.choice((2, 3))
    w0 = min(t0, t1)
    w1 = max(t0, t1)
    p = random.random()
    n = random.choice(range(1, max_op))
    m = random.choice(range(n, max_op))
    s = random.choice(range(m, max_op))
    if random.random() < 0.5:
        e = random.choice(range(1, 2 * max_op))
    else:
        e = random.choice(range(s, 2 * max_op))
    # print(d, w0, w1, e, p)
    graph = Graph(d, w0, w1, e, p)
    # print(n, m, s)
    graph.gen_debug(n, m, s)
    # print(graph.n_inter, graph.n_op, graph.op_num(graph.topological_order))
    # graph.display(detail=False)
    # graph.draw()
    return graph.n_op, graph.op_num(graph.topological_order), graph

def debug():
    '''
    test if n_op is correct.
    '''
    count = 0
    while True:
        op0, op1, graph = auto_easy()
        count += 1
        if count == 100000:
            print("no problem")
            return
        if op0 != op1:
            graph.display()
            graph.draw()
            return



if __name__ == "__main__":
    debug()
    '''p = random.random()
    p = 1/2
    print(p)
    graph = Graph(2, 2, 3, 6, p)'''

    if False:
        graph.gen(n=3, m=4, s=8)
        graph.display(detail=True)
        graph.draw()

    if False:
        for i in range(100000):
            graph.gen(n=3, m=4, s=8)
            if graph.record['inter'] or graph.record['remain']:
                print('Error')
                break
    
    if False:
        count = 0
        for i in range(10000):
            graph.init()
            graph.choose_param(6, 8)
            graph.setup_template()
            # print("finished")
            valid = graph.reasonable_sort(first=-1)
            if not nx.is_directed_acyclic_graph(graph.template):
                print(False)
                break
            if valid:
                count += 1
            if i % 100 == 99:
                print(f"rate = {count / (i + 1)}.")