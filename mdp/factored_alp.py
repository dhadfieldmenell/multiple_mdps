import numpy as np
import util as u
import copy, types
import gurobipy as grb
import IPython as ipy
from collections import defaultdict, deque
from pdb import pm, set_trace
from mdp_base import MultipleMDP
import random
GRB = grb.GRB
eps = 10**-8
MAX_ITER=1000

def factored_alp(mdp):
    """
    takes a a MultipleMDP
    just uses basis functions for now

    returns a vector for the approximate value
    """    
    m = grb.Model()
    w0 = m.addVar(name='w0', obj=1, lb=-GRB.INFINITY)
    comp_weights = {}
    for c, c_mdp in enumerate(mdp.component_mdps):
        cur_weights = []
        for i in range(c_mdp.num_states):
            relevance_weight = 1/float(c_mdp.num_states)
            cur_weights.append(m.addVar(name='{}_{}'.format(c, i), obj=relevance_weight, lb=-GRB.INFINITY))
        comp_weights[c] = cur_weights
    # for each action, add the constraints
    aux_vars = {}
    for a in mdp.actions:        
        # introduce auxilary variables 1 for each component, and a
        # constraint for each state in that component
        action_c, comp_a = mdp.action_to_component(a)
        for c, c_mdp in enumerate(mdp.component_mdps):
            aux_vars[(a, c)] = m.addVar(name='aux_{}_{}'.format(a, c), obj=0, lb=-GRB.INFINITY)
            m.update()
            if action_c == c:
                g_mat = mdp.gamma*c_mdp.transition_dist[comp_a] - np.eye(c_mdp.num_states)
                r_vec = c_mdp.rewards[comp_a]
            else:
                g_mat = mdp.gamma*np.eye(c_mdp.num_states) - np.eye(c_mdp.num_states)
                r_vec = np.zeros(c_mdp.num_states)
            for s in range(c_mdp.num_states):
                rhs_expr = grb.LinExpr(zip(g_mat[s,:], comp_weights[c]))
                rhs_expr += r_vec[s]
                m.addConstr(aux_vars[(a, c)] >= rhs_expr)
        # introduce the joint constraints on all of the auxillary variables
        rhs_expr = grb.quicksum([aux_vars[(a, c)] for c in range(len(mdp.component_mdps))])
        rhs_expr += (mdp.gamma - 1)*w0
        m.addConstr(0 >= rhs_expr)
    m.update()
    m.optimize()
    r_val = []
    for c in comp_weights:
        for w in comp_weights[c]:
            r_val.append(w.X)
    r_val.append(w0.X)
    return r_val

def eval_approx_vf(w, mdp, s):
    if not u.isIterable(s):        
        s = mdp.index_to_factored_state(s)
    r = 0
    offset = 0
    for c, c_mdp in enumerate(mdp.component_mdps):
        r += w[int(offset + s[c])]
        offset += c_mdp.num_states
    r += w[-1]
    return r

def eval_approx_qf(w, mdp, s):
    if not u.isIterable(s):        
        s = mdp.index_to_factored_state(s)
    successors = mdp.successors(s)
    best_a = None
    best_q = -99999
    for a, r, next_states in successors:
        q_a = r
        for (p, n_s) in next_states:
            q_a += mdp.gamma*p*eval_approx_vf(w, mdp, n_s)
        if q_a >= best_q or best_a == None:
            best_a = a
            best_q = q_a
    return best_a, best_q


if __name__ == '__main__':
    import mdp_gen
    params = [mdp_gen.gen_multi_reward_params() for i in range(2)]
    c_mdps = [mdp_gen.multi_reward_grid_mdp(*p) for p in params]
    mdp = MultipleMDP(c_mdps, .9)
    mdp.build_sparse_transitions()
    mdp.frontier_alg()
    w = factored_alp(mdp)
    for i in range(50):
        s = random.randint(0, mdp.num_states)        
        print 'approx', eval_approx_vf(w, mdp, s)
        print eval_approx_qf(w, mdp, s)
        raw_input('go?')

