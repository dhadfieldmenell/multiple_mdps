"""
Functions for acting optimally in multiple MDP -- runs several targeted value iterations 
to prove that one action from the current state is optimal

Maintain UB tree of state space and LB tree of state space
Maintain UB for each state in UB tree and LB for each state in LB tree
Need fast strategy to get unexpanded states associated with top level actions
Pick strategy, then expand some states associated with the upper of lower bound we care about
Add those to the tree, and run a value iteration LP to update estimate
  --prune tree to remove proveably suboptimal constraints
"""

import numpy as np
import util as u
import copy, types
import gurobipy as grb
import IPython as ipy
from collections import defaultdict, deque
from heapq import heappush, heappop
from pdb import set_trace, pm
import cProfile, pstats, StringIO
import sys
import threading, time
#from grb import GRB
GRB = grb.GRB
eps = 10**-8
MAX_ITER=100000

class SearchNode(object):

    """
    node in our search tree
    """

    def __init__(self, state, mdp, use_wi=True, compute_bounds=True):
        self.state = state
        self.mdp = mdp # assumes that mdp.sparse_transitions has already been computed
        self.expanded = False
        self.parent_actions = set()
        # mapping from actions to linear combination of states that represents constraints
        self.children = {} 
        self.use_wi = use_wi
        if compute_bounds:
            if use_wi:
                (self.upper_bound, self.lower_bound) = \
                    self.mdp.whittle_integral(state, compute_lb = True)
            else:
                (self.upper_bound, self.lower_bound) = \
                    self.mdp.singh_cohn_bounds(state)

    # @profile
    def expand(self, known_states):
        """
        computes the successors of this state,
        prunes suboptimal actions

        known_states is a dictionary that maps states we have already expanded to their SearchNodes
        """
        if self.expanded:            
            return set()
        successors = self.mdp.successors(self.state)
        new_nodes = defaultdict(set)
        for (a, r, next_states) in successors:
            next_nodes = []
            for (p, n_s) in next_states:
                # set_trace()
                if n_s not in known_states:
                    known_states[n_s] = SearchNode(n_s, self.mdp, self.use_wi)
                new_nodes[n_s].add(a)
                next_nodes.append((p, known_states[n_s]))
            self.children[a] = (r, next_nodes)
        self.expanded = True
        self.prune()
        kept_nodes = set()
        for new_node in new_nodes:
            for parent_a in new_nodes[new_node]:
                if parent_a in self.children:
                    kept_nodes.add(new_node)
        return kept_nodes

    def best_lb_action(self):
        assert self.expanded
        lbs = [self.action_lb(a) for a in sorted(self.children.keys())]        
        best_i = np.argmax(lbs)
        best_a = sorted(self.children.keys())[best_i]
        self.lower_bound = lbs[best_i]
        return best_a

    def best_ub_action(self, second_best = False):
        assert self.expanded
        ubs = [self.action_ub(a) for a in sorted(self.children.keys())]
        best_i = np.argmin(ubs)
        best_a = sorted(self.children.keys())[best_i]
        self.upper_bound = ubs[best_i]
        return best_a
        
    def prune(self):
        if not self.expanded or not self.children:
            return
        lbs = [self.action_lb(a) for a in sorted(self.children.keys())]
        best_i = np.argmax(lbs)
        # set_trace()
        best_a = sorted(self.children.keys())[best_i]
        max_lb = lbs[best_i]
        for a, (r, transition_dist) in self.children.items():
            if self.action_ub(a) <= max_lb and a != best_a: 
                del self.children[a]        

    def action_lb(self, a):
        (r, transition_dist) = self.children[a]
        return r + self.mdp.gamma*sum(p*next_s.lower_bound for p, next_s in transition_dist)

    def action_ub(self, a):
        (r, transition_dist) = self.children[a]
        return r + self.mdp.gamma*sum(p*next_s.upper_bound for p, next_s in transition_dist)
    # @profile
    def add_vars(self, ub_model, lb_model, ub_lambda_model, lb_lambda_model,
                 ub_vars, lb_vars, ub_lambda_vars, lb_lambda_vars, visited_states, 
                 prune = False, update = True):
        visited_states.add(self.state)
        if prune:
            self.prune()
        if not self.expanded:# ub and lb just equal to whittle integral values
            ub_vars[self.state] = ub_model.addVar(name = str(self.state) + '_ub', obj = 1)
            lb_vars[self.state] = lb_model.addVar(name = str(self.state) + '_lb', obj = 1)
            ub_lambda_vars[self.state] = \
                ub_lambda_model.addVar(name = str(self.state) + '_lambda')

            lb_lambda_vars[self.state] = \
                lb_lambda_model.addVar(name = str(self.state) + '_lambda')
        else:
            if self.state not in ub_vars:
                ub_vars[self.state] = ub_model.addVar(name = str(self.state) + '_ub', obj = 1)
            if self.state not in lb_vars:
                lb_vars[self.state] = lb_model.addVar(name = str(self.state) + '_lb', obj = 1)
            for a in self.children:
                if (self.state, a) not in ub_lambda_vars:
                    ub_lambda_vars[(self.state, a)] = \
                        ub_lambda_model.addVar(name = str((self.state, a)) + '_lambda')
                if (self.state, a) not in lb_lambda_vars:
                    lb_lambda_vars[(self.state, a)] = \
                        lb_lambda_model.addVar(name = str((self.state, a)) + '_lambda')
            for (r, t_dist) in self.children.values(): 
                for (p, child) in t_dist:
                    if child.state not in visited_states:
                        # try:
                        child.add_vars(ub_model, lb_model, ub_lambda_model, lb_lambda_model,
                                       ub_vars, lb_vars, ub_lambda_vars, lb_lambda_vars,
                                       visited_states, prune, False)
                        # except:
                        #     print 'recus
                        #     set_trace()
                            
        if update:
            ub_model.update()
            lb_model.update()
            ub_lambda_model.update()
            lb_lambda_model.update()
        

    def get_lb_transition_mats(self, state_to_ind, ind_to_state, 
                               ub_constrs, is_root = True, ctr = 0):
        state_to_ind[self.state] = ctr
        ind_to_state[ctr] = self.state
        ub_constrs[self.state] = []
        ctr += 1
        if not self.expanded:
            return ctr
        lbs = [self.action_lb(a) for a in sorted(self.children.keys())]
        best_a = sorted(self.children.keys())[np.argmax(lbs)]
        r, t_dist = self.children[best_a]
        for p, child in t_dist:
            if child.state not in state_to_ind:
                ctr = child.get_lb_transition_mats(state_to_ind, ind_to_state, 
                                                   ub_constrs, ctr = ctr, is_root = False)
            ub_constrs[self.state].append((p, child.state))
        if not is_root:
            return ctr    
    # @profile
    def add_constraints(self, ub_model, lb_model, ub_lambda_model, lb_lambda_model, 
                        ub_vars, lb_vars, ub_lambda_vars, lb_lambda_vars,
                        visited_states, update = True, 
                        ub_lambda_constrs = None, lb_lambda_constrs = None):
        visited_states[self.state] = self
        add_lambda_constrs = False
        if ub_lambda_constrs == None:
            add_lambda_constrs = True
            ub_lambda_constrs = defaultdict(list)
            lb_lambda_constrs = defaultdict(list)
        if not self.expanded:
            # if the bounds on this variable are tight enough, 
            # we don't need to add in the constraints

            ub_var = ub_vars[self.state]        
            lb_var = lb_vars[self.state]

            ub_model.addConstr(ub_var == self.upper_bound)
            lb_model.addConstr(lb_var == self.lower_bound)
            update_obj_val(ub_lambda_vars[self.state], self.upper_bound)
            update_obj_val(lb_lambda_vars[self.state], self.lower_bound)
        else:
            ub_var = ub_vars[self.state]        
            lb_var = lb_vars[self.state]

            for (a, (r, transition_dist)) in self.children.iteritems():
                ub_coeffs = [(1, ub_var)]
                lb_coeffs = [(1, lb_var)]
                update_obj_val(ub_lambda_vars[(self.state, a)], r)
                update_obj_val(lb_lambda_vars[(self.state, a)], r)
                for p, next_s in transition_dist:
                    ub_lambda_constrs[next_s.state].append((self.mdp.gamma*p, 
                                                         ub_lambda_vars[(self.state, a)]))
                    lb_lambda_constrs[next_s.state].append((self.mdp.gamma*p, 
                                                         lb_lambda_vars[(self.state, a)]))
                    if next_s.state not in visited_states:
                        next_s.add_constraints(ub_model, lb_model, 
                                               ub_lambda_model, lb_lambda_model, 
                                               ub_vars, lb_vars, 
                                               ub_lambda_vars, lb_lambda_vars,
                                               visited_states, False, 
                                               ub_lambda_constrs, lb_lambda_constrs)
                    ub_coeffs.append((-1 * self.mdp.gamma * p, ub_vars[next_s.state]))
                    lb_coeffs.append((-1 * self.mdp.gamma * p, lb_vars[next_s.state]))
                ub_constr = init_lin_expr(ub_coeffs)
                lb_constr = init_lin_expr(lb_coeffs)
                ub_model.addConstr(ub_constr >= r)
                lb_model.addConstr(lb_constr >= r)
        if add_lambda_constrs:
            lambda_model = {'u': ub_lambda_model, 'l':lb_lambda_model}
            lambda_constrs = {'u': ub_lambda_constrs, 'l':lb_lambda_constrs}
            lambda_vars = {'u': ub_lambda_vars, 'l':lb_lambda_vars}
            
            state_transitions = defaultdict(int)
            num_children = len(self.children.keys())
            for a in self.children:
                for p, next_s in self.children[a][1]:
                    state_transitions[next_s.state] += p/float(num_children)
            def const_val(lu, s):
                if s in state_transitions:
                    return state_transitions[s]
                return 0
            for lu in 'lu':
                for (s, n) in visited_states.iteritems():
                    ## for the upper bound, we may want to make this take into account the 
                    ## second best action, thats what B* says to do....
                    rhs = init_lin_expr(lambda_constrs[lu][s])
                    rhs += const_val(lu, s)
                    if n.expanded:
                        lhs_coeffs = [(1, lambda_vars[lu][(s, a)]) for a in n.children]
                    else:
                        lhs_coeffs = [(1, lambda_vars[lu][s])]
                    lhs = init_lin_expr(lhs_coeffs)
                    lambda_model[lu].addConstr(lhs == rhs)
        if update:
            ub_model.update()
            lb_model.update()
            ub_lambda_model.update()
            lb_lambda_model.update()

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        try:
            return self.state == other.state
        except:
            return False

    def __ne__(self, other):
        try:
            return self.state != other.state
        except:
            return True

    def __repr__(self):
        return str(self.state)
    
    __str__ = __repr__
def update_obj_val(var, v):
    var.obj = v

def init_lin_expr(coeffs):
    if coeffs:
        return grb.LinExpr(coeffs)
    return grb.LinExpr()
# @profile
def expand_loop(frontier, explored_nodes, min_hval, active_states, verbose, max_expand, min_expands=50):
    cur_expansions = 0
    while cur_expansions < max_expand:##
        if len(frontier) == 0:
            if verbose: print '\nexiting b/c no items in frontier'
            break
        hval, state = frontier.pop()
        if state not in active_states:
            continue
        if verbose and not cur_expansions % 50:
            sys.stdout.write('heuristic value:\t{} expanded:\t{}\r'.format(hval, cur_expansions))
            sys.stdout.flush()
        expand_node = explored_nodes[state]
        if expand_node.expanded:
            continue            
        expand_node.expand(explored_nodes)
            #push expansions on with an estimate of their hval
        _, t_dist = expand_node.children[expand_node.best_lb_action()]        
        norm_hval = hval/float(eps + expand_node.upper_bound - expand_node.lower_bound)
        for p, next_s in t_dist:
            new_hval = norm_hval*expand_node.mdp.gamma*p*(next_s.upper_bound - next_s.lower_bound)
            if new_hval > 0: ipy.embed()
            frontier.add(next_s.state, new_hval)
        _, t_dist = expand_node.children[expand_node.best_ub_action()]
        for p, next_s in t_dist:
            new_hval = norm_hval*expand_node.mdp.gamma*p*(next_s.upper_bound - next_s.lower_bound)
            frontier.add(next_s.state, new_hval)
            # bookkeeping
        cur_expansions += 1
        if -1*hval <= min_hval and cur_expansions >= min_expands:
            break
    return cur_expansions

# class ExpansionLoop():

#     def __init__(self, frontier, active_states, explored_nodes):
#         self.frontier = frontier # assumed thread safe
#         self.active_states = active_states
#         self.explored_nodes = explored_nodes
#         self.lock = threading.Lock()
#         self.stop_flag = False
#         self.thread = threading.Thread(target = self.run)
#         self.thread.daemon = True
#         self.num_expansions = 0
#         self.last_hval = 0

#     def check_active(self, state):
#         self.lock.acquire()
#         result = state in self.active_states
#         self.lock.release()
#         return result

#     def set_active_states(self, active_states):
#         self.lock.acquire()
#         self.active_states = active_states
#         self.lock.release()

#     def start(self):
#         self.thread.start()

#     def stop(self):
#         self.stop_flag = True

#     def run(self):
#         SearchNode.lock.acquire()
#         i = 1
#         while True:##
#             SearchNode.lock.release()
#             while len(self.frontier) <= 1:
#                 time.sleep(.00000001)
#             SearchNode.lock.acquire()        
#             # i +=1 
#             # print i
#             hval, state = self.frontier.pop()
#             if state not in self.active_states:
#                 continue
#             expand_node = self.explored_nodes[state]
#             if expand_node.expanded:
#                 continue            
#             new_nodes = expand_node.expand(self.explored_nodes)
#             for n in new_nodes:
#                 if n not in self.active_states:
#                     self.active_states[n] = 0
#             #push expansions on with an estimate of their hval
#             _, t_dist = expand_node.children[expand_node.best_lb_action()]
#             norm_hval = hval/float(eps + expand_node.upper_bound - expand_node.lower_bound)
#             for p, next_s in t_dist:
#                 new_hval = norm_hval*expand_node.mdp.gamma*p*(next_s.upper_bound - next_s.lower_bound)
#                 self.frontier.add(next_s.state, new_hval)
#             _, t_dist = expand_node.children[expand_node.best_ub_action()]
#             for p, next_s in t_dist:
#                 new_hval = norm_hval*expand_node.mdp.gamma*p*(next_s.upper_bound - next_s.lower_bound)
#                 self.frontier.add(next_s.state, new_hval)
#             # bookkeeping
#             self.num_expansions += 1
#             self.last_hval = hval
#             if self.stop_flag:
#                 SearchNode.lock.release()
#                 return

# @profile
def local_search_vi(mdp, start_state, tolerance = eps, expand_rate = 100, verbose = False,
                    semi_verbose = False, max_expand = 100, explored_nodes = None, use_wi=True,
                    min_hval_decrease_rate = None):


    if explored_nodes == None:
        root = SearchNode(start_state, mdp, use_wi=use_wi)
        # root = SearchNode(start_state, mdp, True)
        explored_nodes = {start_state: root}

    else:
        root = explored_nodes[start_state]

    def converged():
        ubs = [root.action_ub(a) for a in sorted(root.children.keys())]
        lbs = [root.action_lb(a) for a in sorted(root.children.keys())]
        best_lb_i = np.argmax(lbs)
        best_lb = lbs[best_lb_i]
        if semi_verbose:
            print 'best value', best_lb
            print 'root children:\t', sorted(root.children.keys())
        if verbose:
            print 'upper bounds:\t', ubs
            print 'lower bounds:\t', lbs
        potential_subopt = 0
        for j in range(len(ubs)):
            if j != best_lb_i:
                potential_subopt = max(potential_subopt, ubs[j] - best_lb)
        if potential_subopt >= tolerance:
            if verbose or semi_verbose: 
                print "\nnot converged potential suboptimality of", potential_subopt
            return False, potential_subopt
        return True, potential_subopt

    init_lb = root.lower_bound
    init_ub = root.upper_bound
    greedy_action = mdp.greedy_a(start_state)


    frontier = u.HeapPQ()
    if not root.expanded:
        new_nodes = root.expand(explored_nodes)
        active_states = {}
        for n in new_nodes:
            active_states[n] = n

        for n in new_nodes:
            frontier.add(n)

    active_states = explored_nodes


    num_expanded = 1
    prev_print_val = 0
    min_hval = 1e-2
    if min_hval_decrease_rate is None:
        min_hval_decrease_rate = root.mdp.gamma*0.75
    while not converged()[0] and num_expanded < max_expand:
        if semi_verbose or verbose: print 'min_hval:\t', min_hval
        # expand a bunch of nodes
        #so we expand more next time
        max_loop_expand = max_expand - num_expanded
        new_expansions = expand_loop(frontier, explored_nodes, min_hval, active_states, semi_verbose, max_loop_expand)
        num_expanded += new_expansions
        min_hval *= min_hval_decrease_rate
        # if min_hval < 10**-20:
        #     break
        if verbose or semi_verbose:
            print '\ncurrent nodes expanded:\t', new_expansions
            print "total nodes expanded:\t", num_expanded
            print "lower bound value:\t{}\twithout search:\t{}".format(root.lower_bound, init_lb)
            print "best action:\t{}\tgreedy action:\t{}".format(root.best_lb_action(), greedy_action)

        if num_expanded - prev_print_val > 500:
            print 'num expansions:\t{}'.format(num_expanded)
            prev_print_val = num_expanded
        # run bounded value iteration LP
        m_ub = grb.Model()# value iteration on the upper bound
        m_lb = grb.Model()# value iteration on the lower bound
        m_ub_lambda = grb.Model()# dual on bound LP, tweaked to measure potential effect of 
        m_lb_lambda = grb.Model()# expanding a node on Q estimates
        ub_vars = {}
        lb_vars = {}
        ub_lambda_vars = {}
        lb_lambda_vars = {}
        active_states = {}
        #build the models by pruning and walking the tree
        root.prune()
        root.add_vars(m_ub, m_lb, m_ub_lambda, m_lb_lambda,
                      ub_vars, lb_vars, ub_lambda_vars, lb_lambda_vars, set())
        root.add_constraints(m_ub, m_lb, m_ub_lambda, m_lb_lambda,
                             ub_vars, lb_vars, ub_lambda_vars, lb_lambda_vars, active_states)
        # optimize models
        for m in (m_ub, m_lb, m_ub_lambda, m_lb_lambda):
            if not verbose:
                m.setParam('OutputFlag', False)
            m.optimize()
            if m.status == GRB.status.INFEASIBLE:
                # sometimes the presolver makes us think this is infeasible when it isn't...
                m.setParam(GRB.param.Presolve, 0)
                if verbose: print "infeasible model, turning off presolve"
                m.optimize()
        if m_ub.status == GRB.status.INFEASIBLE:
            print 'ub model infeasible!!'
            print 'test'
            raise Exception
        if m_lb.status == GRB.status.INFEASIBLE:
            print 'lb model infeasible!!'
            print 'test'
            raise Exception
        reachable_states = 0
        to_delete = set()
        #clean our explored nodes, push potential expansions onto heap
        for (state, node) in active_states.iteritems():
            ub = ub_vars[state].X
            lb = lb_vars[state].X
            if not node.expanded:
                h_val = max(lb_lambda_vars[state].X, ub_lambda_vars[state].X)*(ub - lb) 
                # maximum increase in lb from this var changing
            if ub < lb:
                if verbose:
                    print 'upper bound less than lower bound, difference', ub - lb
                ub = lb
            # if ub < node.lower_bound - eps or lb > node.upper_bound + eps:
            #     print 'Node Values Outside Proven Range'
            #     ipy.embed()
            node.upper_bound = ub
            node.lower_bound = lb
            if not node.expanded:
                frontier.add(state, -1*h_val)
            reachable_states += 1
        if semi_verbose or verbose: print 'reachable states:\t', reachable_states
    _converged, sub_optimality = converged()
    ubs = [root.action_ub(a) for a in sorted(root.children.keys())]
    lbs = [root.action_lb(a) for a in sorted(root.children.keys())]

    print "search returned init action:\t{}\twithout search:\t{}".format(root.best_lb_action(), greedy_action)
    print "converged?:\t{}\tafter {} expansions".format(_converged, num_expanded)
    print "lower bound value:\t{}\twithout search:\t{}".format(root.lower_bound, init_lb)
    print "upper bound value:\t{}\twithout search:\t{}".format(root.upper_bound, init_ub)

    return root.best_lb_action(), num_expanded, sub_optimality

def lrtdp(mdp, start_state, tolerance=eps):
    
    root = SearchNode(start_state, mdp, use_wi=use_wi)
    explored_nodes = {start_state: root}
