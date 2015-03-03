from __future__ import division

import numpy as np

import sys

eps = 1e-8

class SearchNode(object):

    NODE_REGISTER = {}
    USE_WI = False

    def __init__(self, state, mdp):
        self.state = state
        self.mdp = mdp
        if SearchNode.USE_WI:
            self.value = mdp.whittle_integral(state)
        else:
            self.value = mdp.singh_cohn_bounds(state)[0]

        self.successors = dict([(x[0], (x[1], x[2])) for x in self.mdp.successors(self.state)])
        self.solved = False
        SearchNode.NODE_REGISTER[state] = self    

    def qvalue(self, a):
        Q, next_states = self.successors[a]
        for p , n_s in next_states:
            if n_s not in SearchNode.NODE_REGISTER:
                n_s = SearchNode(n_s, self.mdp)
            else:
                n_s = SearchNode.NODE_REGISTER[n_s]
            Q += self.mdp.gamma * p * n_s.value
        return Q

    def greedy(self):
        best_q = -np.inf
        for a in self.successors:
            cur_q = self.qvalue(a)
            if cur_q > best_q:
                best_q = cur_q
                best_a = a
        return best_a

    def update(self):
        a = self.greedy()
        self.value = self.qvalue(a)

    def sample_next(self, a):
        _, next_states = self.successors[a]
        next_states, probs = ([self.mdp.factored_state_to_index(x[1]) for x in next_states], 
                              [x[0] for x in next_states])
        next_s_id = np.random.choice(next_states, 1, probs)[0]        
        next_s = self.mdp.index_to_factored_state(next_s_id)
        if next_s not in SearchNode.NODE_REGISTER:
            return SearchNode(next_s, self.mdp)
        return SearchNode.NODE_REGISTER[next_s]

    def residual(self):
        a = self.greedy()
        return np.abs(self.value - self.qvalue(a))  

    def check_solved(self, tol):
        converged = True
        _open = []
        _closed = []
        if not self.solved: _open.append(self)
        while _open:
            s = _open.pop()
            _closed.append(s)
            if s.residual() > tol:
                converged = False
                continue
            a = s.greedy()
            _, next_states = s.successors[a]
            for _, s_next in next_states:
                s_next = SearchNode.NODE_REGISTER[s_next]
                if not s_next.solved and (
                    not (s_next in _open or s_next in _closed)):
                    _open.append(s_next)
        if converged:
            # print "found converged states"
            for s in _closed:
                s.solved=True
        else:
            for s in _closed:
                s.update()
        return converged

def lrtdp(mdp, start_state, tol=eps, use_wi=False, max_iter=100):
    
    sim_len = int(np.ceil(np.log(tol) / np.log(mdp.gamma)))

    SearchNode.USE_WI=use_wi
    SearchNode.NODE_REGISTER={}
    
    root = SearchNode(start_state, mdp)
    for _ in range(max_iter):
        if root.solved:
            break
        sys.stdout.write("Root Residual:\t{}\r".format(root.residual()))
        sys.stdout.flush()
        visited = []
        s = root
        for i in range(sim_len):
            visited.append(s)
            a = s.greedy()
            s = s.sample_next(a)
            if s.solved:
                break
        while visited:
            s = visited[-1]
            visited = visited[:-1]
            if not s.check_solved(tol):
                break

    return root.solved, root.greedy()
