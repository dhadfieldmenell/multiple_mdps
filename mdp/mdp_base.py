import numpy as np
import util
import copy, types
import gurobipy as grb
import IPython as ipy
from collections import defaultdict, deque
from pdb import pm, set_trace
import random, time
from scipy import linalg
from scipy.sparse import linalg
import scipy as sp
GRB = grb.GRB
eps = 10**-8
MAX_ITER=100000

class MDP(object):
    
    def __init__(self, actions, num_states, T, R, gamma):
        self.actions = copy.copy(actions)
        self.transition_dist = copy.deepcopy(T)
        self.num_states = num_states
        self.rewards = copy.deepcopy(R)
        self.gamma = gamma
        self.value_fn  = None
        self.opt_policy = None
        self.gittins_indices = None
        self.phi_primes = None
        self.log_phi_primes = None
        self.phi_vals = None
        self.m_vals = None
        self.max_m_val = None
        self._m_query_vals = None
        self._m_query_inds = None
        self.phi_prime_query_cache = {}
        self.ret_phi_val_cache = {}
        self.sparse_transitions = defaultdict(list) # mapping from a state, action to a pairing of probabilities and states
        self.sparse_t_computed = False
        self.state_to_actions = defaultdict(set)
        self.wc_violations = set()
        self.mrp_cache = {}        

    @property
    def m_query_vals(self):
        return self._m_query_vals

    @m_query_vals.setter
    def m_query_vals(self, value):
        self.phi_prime_query_cache = {}
        self._m_query_vals = sorted(value[:])
        self._m_query_inds = np.zeros(value.shape)
        cur_ind = 0
        (min_m, max_m) = self.m_vals[cur_ind]
        for (i, m) in enumerate(self._m_query_vals):
            if m >= self.max_m_val:
                #length is always the same so just use length from 0 state
                self._m_query_inds[i:] = len(self.phi_primes[0]) - 1
                break
            while m >= max_m:
                cur_ind += 1
                (min_m, max_m) = self.m_vals[cur_ind]
            self._m_query_inds[i] = cur_ind

        self._m_query_inds = self._m_query_inds.astype(int)

    @property
    def m_query_inds(self):
        return self._m_query_inds

    @m_query_inds.setter
    def m_query_inds(self):
        raise Exception, "Can not directly set m_query_inds"

    def build_sparse_transitions(self):
        if self.sparse_t_computed:
            return
        # if self.state_to_actions == None:
        #     raise Exception, 'state to actions must be filled out to compute sparse transitions'
        for a in self.actions:
            non_zero = np.transpose(np.nonzero(self.transition_dist[a]))
            for (s1, s2) in non_zero:
                # if a not in self.state_to_actions[s1]:
                #     continue
                self.sparse_transitions[(s1, a)].append((self.transition_dist[a][s1, s2], s2))
                self.state_to_actions[s1] |= set([a])
        self.sparse_t_computed = True

    def ret_val_fn(self, m, s = None):
        if m not in self.ret_phi_val_cache:
            m_ind = self.get_m_ind(m)
            local_vals = self.phi_vals[m_ind][1]
            ret_rewards = self.phi_primes[:, m_ind][1]*m
            self.ret_phi_val_cache[m] = local_vals + ret_rewards
        if s is not None:
            return self.ret_phi_val_cache[m][s]
        return self.ret_phi_val_cache[m]

    def get_retirement_mdp(self, ret_rewards):
        new_num_states = self.num_states + 1
        new_T = {}
        new_R = {}
        ##add new state, copy over action transitions
        for a, T_a in self.transition_dist.iteritems():
            new_T_a = np.zeros((new_num_states, new_num_states))
            new_T_a[:self.num_states, :self.num_states] = T_a[:,:]
            new_T_a[new_num_states-1, new_num_states-1] = 1
            new_T[a] = new_T_a
            new_R_a = np.zeros(new_num_states)
            new_R_a[:self.num_states] = self.rewards[a][:]
            new_R_a[-1] = 0
            new_R[a] = new_R_a
        new_T_ret = np.zeros((new_num_states, new_num_states))
        new_T_ret[:,-1] = 1
        new_T['retire'] = new_T_ret
        new_R['retire'] = np.zeros(new_num_states)
        new_R['retire'][:self.num_states] = ret_rewards[:]
        actions = self.actions[:]
        actions.append('retire')
        return MDP(actions, new_num_states, new_T, new_R, self.gamma)

    def to_retirement_mrp(self, ret_rewards):
        """
        takes as an argument a set of retirement rewards for each state
        computes an optimal policy wrt those rewards
        returns an MRP that is this MDP restricted to those actions
        """
        ret_mdp = self.get_retirement_mdp(ret_rewards)
        values = ret_mdp.compute_value_fn()
        opt_pol = [util.argmax(self.actions, 
                            lambda x: self.eval_action(s, x, values[:-1]))#ignore the retirement state
                   for s in range(self.num_states)]
        return self.to_mrp(opt_policy = opt_pol)        
        

    def to_mrp(self, m = 0, compute_values = None, opt_policy = None):
        # computes the optimal policy, then builds an mdp
        # that only has the optimal policy as an option
        m_ind = None
        if opt_policy == None:
            if compute_values != None:
                policy_vals = [compute_values(s) for s in range(self.num_states)]
                opt_policy = [util.argmax(self.actions, 
                                       lambda x: self.eval_action(s, x, policy_vals))
                              for s in range(self.num_states)]
            else:
                m_ind = self.get_m_ind(m)
                if m_ind in self.mrp_cache:
                    return copy.deepcopy(self.mrp_cache[m_ind])
                opt_policy = self.optimal_policy(m)
        T_a = np.zeros((self.num_states, self.num_states))
        R_a = np.zeros(self.num_states)
        for s in range(self.num_states):
            best_a = opt_policy[s]
            T_a[s, :] = self.transition_dist[best_a][s,:]
            R_a[s] = self.rewards[best_a][s]
        act_name = 'opt_a'
        T = {act_name: T_a}
        R = {act_name: R_a}        
        mrp = MDP([act_name], self.num_states, T, R, self.gamma)
        if m_ind is not None:
            self.mrp_cache[m_ind] = mrp
        # mrp.frontier_alg()
        return copy.deepcopy(mrp)
    # @profile
    def successors(self, state, action):
        if not self.sparse_t_computed:
            self.build_sparse_transitions()
        return self.sparse_transitions[(state, action)]

    # @profile
    def frontier_alg(self, eta_max = 20, debug = False):
        """
        Implements the frontier algorithm for an MDP from 
        the Brown and Smith paper
        
       Looks at the value function for various values of the
        retirement value and computes:
             - retirement values where the policy changes
             - derivatives of retirement value as a fn of M and state      
        Maintains an eta factorization of the basis matrix and uses an LU decomposition 
        reasonably fast
        """
        # if self.m_vals != None:
        #     return
        M_vals = []
        ## initial value is R*/(1-gamma) b/c that corresponds to getting maximum
        ## reward forever
        max_reward = np.max([np.max(self.rewards[k]) for k in self.rewards])
        M_vals.append(np.exp(np.log(max_reward) - np.log(1-self.gamma)))
        m_j = M_vals[0]
        self.gittins_indices = np.zeros(self.num_states)

        ##useful lists to calculate indices
        aug_actions = self.actions + ['Retire']
        ind_to_sa = [(s, a) for s in range(self.num_states) for a in aug_actions]
        sa_to_ind = dict((v, i) for i, v in enumerate(ind_to_sa))
        s_to_a = ['Retire' for s in range(self.num_states)]

        exp_rewards = np.zeros((self.num_states, len(aug_actions)))
        for (i, a) in enumerate(self.actions):
            exp_rewards[:,i] = self.rewards[a]
        #c from the paper
        exp_rewards = exp_rewards.flatten()
        retire_indicator = np.zeros((self.num_states, len(aug_actions)))
        retire_indicator[:, len(self.actions)] = 1
        # d from the paper
        retire_indicator = retire_indicator.flatten()

        #A from paper
        A_T = np.zeros((len(aug_actions)*self.num_states, self.num_states))

        for i, (s, a) in enumerate(ind_to_sa):
            A_T[i, s] = 1
            ## explicitly compute successors if this is not retire
            if a == 'Retire': continue
            next_states = np.nonzero(self.transition_dist[a][s, :])[0]
            for next_s in next_states:
                ## other side of bellman eqs 
                A_T[i, next_s] -= self.gamma*self.transition_dist[a][s, next_s]
        A_lil_T = sp.sparse.lil_matrix(A_T)
        A_csr_T = A_lil_T.tocsr()
        A = A_T.T
        #initially retirement is optimal
        
        b_lil = sp.sparse.eye(self.num_states, format='lil')

        c_b = np.zeros(self.num_states)
        d_b = np.ones(self.num_states)
        phi_primes = []
        phi_vals = []
        eta_list = []
        lu_solve = sp.sparse.linalg.factorized(b_lil)
        state_action_map = {}
        for s in range(self.num_states):
            state_action_map[s] = set()
        start = time.time()
        for j in range(MAX_ITER):#so this doesn't run forever
            #calculate reduced costs            
            if len(eta_list) >= eta_max:
                # for eta, eta_inv, k in eta_list:
                #     util.eta_mult_mat_inplace(eta, k, b_lil)
                eta_list = []
                lu_solve = sp.sparse.linalg.factorized(b_lil)

            l_c = util.eta_solve(c_b, lu_solve, eta_list)
            # l_c_tmp = sp.sparse.linalg.spsolve(b_lil, c_b)
            l_d = util.eta_solve(d_b, lu_solve, eta_list)
            # l_d_tmp = sp.sparse.linalg.spsolve(b_lil, d_b)

            c_bar = exp_rewards - A_csr_T.dot(l_c)
            d_bar = retire_indicator - A_csr_T.dot(l_d)

            if np.all(d_bar >= 0):
                # print 'all d_bar >=0'
                # set_trace()
                phi_primes.append(((0, M_vals[j]), np.zeros(l_d.shape)))
                phi_vals.append(((0, M_vals[j]), l_c))
                M_vals.append(0)
                break                
            
            # computing argmax and max explicity to take into account constraints
            m_j = None            
            # I = set()
            ratio = np.zeros(len(c_bar))
            # ratio_tmp = np.zeros(len(c_bar))
            non_zero_inds = (d_bar <= -1*eps)
            ratio[non_zero_inds] = np.divide(c_bar[non_zero_inds], -1*d_bar[non_zero_inds])
            # ratio[non_zero_inds] = np.exp(np.log(c_bar[non_zero_inds]) - np.log(-1*d_bar[non_zero_inds]))
            # assert np.allclose(ratio, ratio_tmp)

            m_j = np.max(ratio)
            I =  np.nonzero(ratio == m_j)[0]
            m_j = max(m_j, 0)
            if debug:
                set_trace()
            phi_primes.append(((m_j, M_vals[j]), l_d)) #derivatives of value fn wrt M for those values
            phi_vals.append(((m_j, M_vals[j]), l_c)) #value fn from this MDP w retirement M
            if m_j <=0 :
                M_vals.append(0)
                # print 'm_j <= 0'
                # set_trace()
                break
            M_vals.append(m_j)
            marked_states = set()
            ## keep track of states we've already seen. if there are 
            ## two actions with the same value we will always pick the
            ## one first in that ordering. 
            # set_trace()
            for sa_ind in sorted(I):
                (s, a) = ind_to_sa[sa_ind]
                if s in marked_states:
                    continue
                marked_states.add(s)
                state_action_map[s].add(a)
                self.gittins_indices[s] = max(m_j, self.gittins_indices[s])
                cur_a = s_to_a[s]
                cur_sa_ind = sa_to_ind[(s, cur_a)]
                # compute eta matrix
                trans_solve = lambda x: lu_solve(x, trans='T')
                d = util.eta_solve(A[:, sa_ind], trans_solve, eta_list, transpose=True)
                ## do divisions in log space to make things more stable
                # d_inv = -1* np.exp(np.log(d) - np.log(d[s]))
                d_inv = -np.divide(d, d[s])
                # d_inv[s] = np.exp(-1*np.log(d[s]))
                d_inv[s] = np.divide(1, d[s])

                eta_list.append((d, d_inv, s))
                b_lil.rows[s] = A_lil_T.rows[sa_ind]
                b_lil.data[s] = A_lil_T.data[sa_ind]
                c_b[s] = exp_rewards[sa_ind]
                d_b[s] = retire_indicator[sa_ind]
                s_to_a[s] = a
        if j == MAX_ITER-1:
            print 'Warning, WI calculation did not run to convergence, max iter reached'
        self.state_to_actions = {}
        self.wc_violations = set()
        self.wc_actions = {}
        num_pruned = 0
        for s in range(self.num_states):
            if len(state_action_map[s]) <= 1:
                # if there's just one then we can prune everything else

                num_pruned += 1
                # self.state_to_actions[s] = state_action_map[s]
            else:
                self.wc_actions[s] = state_action_map[s]
                self.wc_violations.add(s)
            self.state_to_actions[s] = set(self.actions[:])
        self.wc_violations = list(self.wc_violations)
        wi_frac = num_pruned/float(self.num_states)
        # print '{} of {} states satisfy the WC'.format(num_pruned, self.num_states)
        self.build_sparse_transitions()
        # set_trace()
        self.max_m_val = np.max(M_vals)
        phi_primes.reverse() # so that the values are in ascending order
        self.m_vals = []
        self.phi_primes = np.zeros((self.num_states, len(phi_primes)))
        for i, ((min_m, max_m), cur_vals) in enumerate(phi_primes):
            self.m_vals.append((min_m, max_m))
            self.phi_primes[:, i] = np.maximum(cur_vals, 0)
        self.log_phi_primes = np.log(self.phi_primes)
        # for s in range(self.num_states):
        #     self.log_phi_primes[s] = np.log(self.phi_primes[s, :])
        self.phi_vals = phi_vals
        self.phi_vals.reverse()
        return wi_frac, time.time() - start, j
    # @profile
    def all_phi_prime(self, s):
        """
        return a vector of phi_prime values at s for each m in self.m_query_vals
        """
        return self.phi_primes[s][self.m_query_inds]

    def get_m_ind(self, M):
        min_ind = 0
        max_ind = len(self.m_vals) - 1
        mid_ind = max_ind/2
        (mid_min, mid_max) = self.m_vals[mid_ind]
        while M >= mid_max or M < mid_min:
            if M >= mid_max:
                min_ind = mid_ind
            elif M < mid_min:
                max_ind = mid_ind
            mid_ind = (max_ind + min_ind)/2
            (mid_min, mid_max) = self.m_vals[mid_ind]
        return mid_ind

    # def phi_prime(self, s, M):
    #     if not self.phi_primes:
    #         self.frontier_alg()
    #     if M > self.max_m_val:
    #         return 1
    #     ## binary search to find location in list
    #     m_ind = self.get_m_ind(M)
    #     return self.phi_primes[m_ind][1][s]

    def gittins_index(self, s):
        if not self.gittins_indices:
            self.frontier_alg()
        return self.gittins_indices[s]

    def compute_value_fn(self):
        m = grb.Model()
        model_vars = [m.addVar(lb = -1*GRB.INFINITY, name=str(i), obj = 1) 
                      for i in range(self.num_states)]
        m.update()
        for cur_state_i in range(self.num_states):
            cur_state_v = model_vars[cur_state_i]
            for a in self.actions:
                cur_reward = self.rewards[a][cur_state_i]
                T = self.transition_dist[a]
                successors = self.successors(cur_state_i, a)
                rhs_coeffs = []
                for p, next_s in successors:
                    rhs_coeffs.append((p, model_vars[next_s]))                
                rhs = grb.LinExpr(rhs_coeffs)
                rhs *= self.gamma
                rhs += cur_reward
                m.addConstr(cur_state_v >= rhs)
        m.setParam('OutputFlag', False)
        m.update()
        m.optimize()
        self.value_fn = [x.X for x in model_vars]
        return self.value_fn
        
    
    def eval_action(self, s, a, values):
        ## values is an array of values for the states
        return np.dot(self.transition_dist[a][s], values)
    
    def Q_fn(self, s, a = None):
        if not self.value_fn:
            self.compute_value_fn()
        if not a:
            a = self.opt_policy[s]
        return self.eval_action(s, a, self.value_fn)

    def optimal_policy(self, m=0):
        values = self.ret_val_fn(m)
        self.opt_policy = [util.argmax(self.actions, 
                                    lambda x: self.eval_action(s, x, values))
                                    for s in range(self.num_states)]
        return self.opt_policy                                          

class MultipleMDP(object):
    
    def __init__(self, component_mdps, gamma):
        self.component_mdps = copy.deepcopy(component_mdps)
        self.actions = util.smash([[self.action_str(a, i) for a in component_mdps[i].actions] 
                                for i in range(len(component_mdps))])
        self.num_states = 1
        for c_mdp in component_mdps:
            self.num_states = self.num_states*c_mdp.num_states
            c_mdp.gamma = gamma
        self.num_components = len(component_mdps)
        self.gamma = gamma
        self.product_mdp = None
        self.gittins_indices = None
        self.value_fn = None
        self.lb_mdp = None
        self.init_int_val = None

    def get_actions(self, state):
        """
        returns the actions that exit this state with non-zero probability
        """
        next_actions = []
        for (c_num, c_state) in enumerate(state):
            next_actions.extend(self.action_str(a, c_num) for a in 
                                     self.component_mdps[c_num].state_to_actions[c_state])
        return next_actions

    def get_rand_state(self, wc_violations_only=False):
        if wc_violations_only:
            state = []
            for j, c_mdp in enumerate(self.component_mdps):
                if c_mdp.wc_violations:
                    state.append(random.choice(c_mdp.wc_violations))
                else:
                    state.append(random.randint(0, c_mdp.num_states))
            return state
        else:
            return self.index_to_factored_state(random.randint(0, self.num_states))
                
            
        

    def successors(self, state, action = None):
        ## returns a dictionary that maps actions to next states, probabilities
        state = tuple(state)
        if not action:
            next_states = []
            for a in self.get_actions(state):
                next_states.append(self.successors(state, a))
            return next_states
        # if an action is specified, just return those results
        (c, a) = self.action_to_component(action)
        c_mdp = self.component_mdps[c]
        c_s = state[c]
        next_comp_states = c_mdp.successors(c_s, a)
        next_states = [(p, tuple(state[:c] + (n_c_s,) + state[c+1:])) for (p, n_c_s) 
                                 in next_comp_states]
        return (action,) +  util.tuplify([c_mdp.rewards[a][c_s], next_states])

    def average_transition_rate(self):
        avgs = []
        for c_mdp in self.component_mdps:
            for a in c_mdp.actions:
                avgs.append(np.mean(c_mdp.transition_dist[a][c_mdp.transition_dist[a] != 0]))
        return np.mean(avgs)

    def build_sparse_transitions(self):
        for c_mdp in self.component_mdps:
            c_mdp.build_sparse_transitions()            

    def compute_lb_mdp(self, eta_max=20, state = None):
        # set_trace()
        self.lb_mdp = MultipleMDP([c_mdp.to_mrp() for c_mdp in self.component_mdps],
                                  self.gamma)
        self.lb_mdp.frontier_alg()
        return self.lb_mdp

    def frontier_alg(self, eta_max=20):
        wi_fracs = []
        self.gittins_indices = set()
        for c_mdp in self.component_mdps:
            # if not c_mdp.phi_primes:
            wi_fracs.append(c_mdp.frontier_alg(eta_max=eta_max)[0])
            self.gittins_indices |= set(np.asarray(c_mdp.m_vals).flatten())
        self.gittins_indices = np.asarray(sorted(self.gittins_indices))
        for c_mdp in self.component_mdps:
            c_mdp.m_query_vals = self.gittins_indices[:-1]
        self.init_int_val = np.log(self.gittins_indices[1:] - self.gittins_indices[:-1])
        print wi_fracs
        return wi_fracs
    # @profile
    def whittle_integral(self, state, compute_lb = False, recompute_lb_mdp = False):
        if not util.isIterable(state):
            # switch to factored representation
            state = self.index_to_factored_state(state)
        # compute integral
        int_val = self.init_int_val.copy()
        for (j, c_mdp) in enumerate(self.component_mdps):
            int_val += c_mdp.log_phi_primes[state[j]][c_mdp.m_query_inds]

        wi_val = self.gittins_indices[-1] - np.sum(np.exp(int_val))
        # if check_ubs and abs(wi_val - ub_phis['val']) > 1e-6:
        #     ipy.embed()
        if np.isnan(wi_val):
            print 'NAN'
            raise Exception, 'NAN encountered in Whittle Integral'
        # n_pruned = 0
        # for j, c_mdp in enumerate(self.component_mdps):
        #     n_pruned += c_mdp.ret_val_fn(wi_val, state[j]) == wi_val
        # print 'pruned {}'.format(n_pruned)
        if compute_lb:
            lb_wi = self.lb_mdp.whittle_integral(state, compute_lb = False)
            if lb_wi > wi_val:
                if lb_wi - wi_val > 1e-4:
                    set_trace()
                wi_val = lb_wi
            # print state, wi_val, lb_wi
            return (wi_val, lb_wi)
        return wi_val

    def greedy_a(self, state):
        if not util.isIterable(state):
            # switch to factored representation
            state = self.index_to_factored_state(state)
        best_gi = -np.inf
        for i, (m_i, s_i) in enumerate(zip(self.lb_mdp.component_mdps, state)):
            if m_i.gittins_indices[s_i] > best_gi:
                best_a = self.component_mdps[i].opt_policy[s_i]
                best_i = i
                best_gi = m_i.gittins_indices[s_i]
        return "{}_{}".format(best_a, best_i)


    def singh_cohn_bounds(self, state):
        # set_trace()
        if not util.isIterable(state):
            # switch to factored representation
            state = self.index_to_factored_state(state)
        vals = np.zeros(len(self.component_mdps))
        for (j, c_mdp) in enumerate(self.component_mdps):
            vals[j] = c_mdp.ret_val_fn(0, state[j])
        return (np.sum(vals), np.max(vals))

    def factored_state_to_index(self, state):
        s_i = 0
        for i in range(len(self.component_mdps)):
            s_i *= self.component_mdps[i].num_states
            s_i += state[i]
        return int(s_i)

    def index_to_factored_state(self, s_i):
        s_result = np.zeros(self.num_components, dtype=np.int)
        for i in reversed(range(len(self.component_mdps))):
            num_states = self.component_mdps[i].num_states
            s_i_new = s_i/num_states
            s_result[i] = s_i - s_i_new*num_states
            s_i = s_i_new
        return tuple(s_result.tolist())

    def action_to_component(self, a):
        a_split = a.split('_')
        return (int(a_split[1]), a_split[0])

    def action_str(self, a, i):
        return a + '_' + str(i)

    def transition_dist(self, s1, a, s2):
        (comp_num, orig_a) = self.action_to_component(a)
        factored_s1 = self.index_to_factored_state(s1)
        factored_s2 = self.index_to_factored_state(s2)
        proj_s1 = factored_s1[i]
        proj_s2 = factored_s2[i]
        return self.component_mdps[i].transition_dist[orig_a][proj_s1][proj_s2]
        
    def compute_product_mdp(self):
        T = {}
        R = {}
        for a in self.actions:
            (comp_num, orig_a) = self.action_to_component(a)
            T[a] = np.zeros((self.num_states, self.num_states))
            R[a] = np.zeros(self.num_states)
            comp_T = self.component_mdps[comp_num].transition_dist[orig_a]
            for s1 in range(self.num_states):
                factored_s1 = self.index_to_factored_state(s1)
                sub_s1 = factored_s1[comp_num]
                for sub_s2 in range(self.component_mdps[comp_num].num_states):
                    factored_s2 = copy.copy(factored_s1)
                    factored_s2[comp_num] = sub_s2
                    s2 = self.factored_state_to_index(factored_s2)
                    T[a][s1][s2] = comp_T[sub_s1][sub_s2]                
                R[a][s1] = self.component_mdps[comp_num].rewards[orig_a][sub_s1]
        self.product_mdp = MDP(self.actions, self.num_states, T, R, self.gamma)
        return self.product_mdp

    def compute_value_fn(self, verbose=False, max_time = 600):
        start = time.time()
        m = grb.Model()
        model_vars = [m.addVar(lb = -1*GRB.INFINITY, name=str(i), obj = 1) for i in range(self.num_states)]
        m.update()
        for s1 in range(self.num_states):
            if time.time() - start > max_time:
                raise Exception
            if verbose and s1%100 == 0:
                print s1
            cur_state_v = model_vars[s1]
            factored_s1 = self.index_to_factored_state(s1)
            successors = self.successors(factored_s1)
            for (a, r, next_states) in successors:
                rhs_coeffs = []
                for p, next_s in next_states:
                    n_s = self.factored_state_to_index(next_s)
                    rhs_coeffs.append((p, model_vars[n_s]))
                rhs_expr = grb.LinExpr(rhs_coeffs)
                rhs_expr *= self.gamma
                rhs_expr += r
                m.addConstr(cur_state_v >= rhs_expr)
        m.update()
        if not verbose:
            m.setParam('OutputFlag', False)
        m.optimize()
        self.value_fn = [x.X for x in model_vars]
        return self.value_fn

    def compare_value_fn_and_wi(self, state):
        s_ind = self.factored_state_to_index(state)
        if not self.value_fn:
            self.compute_value_fn()
        return (self.value_fn[s_ind], self.whittle_integral(state))

    def compare_all_states(self, verbose=False, eta_max=20):
        self.build_sparse_transitions()
        if not self.value_fn:
            self.compute_value_fn(verbose=verbose)
        self.frontier_alg(eta_max=eta_max)
        self.compute_lb_mdp(eta_max=1)
        differences = []
        for s_ind in range(self.num_states):
            wi_u, wi_l = self.whittle_integral(s_ind, compute_lb=True)
            if self.value_fn[s_ind] > wi_u + eps:
                print 'upper bound too small, real: {},ub: {}'.format(self.value_fn[s_ind], wi_u)
                raise Exception
            if self.value_fn[s_ind] < wi_l - eps:
                print 'lower bound too large, real: {}, lb: {}'.format(self.value_fn[s_ind], wi_l)
                raise Exception
            if not util.nearlyEqual(wi_u, wi_l):
                differences.append((wi_u - wi_l, wi_u, wi_l, self.value_fn[s_ind], s_ind))
                if verbose and not len(differences) % 30: print s_ind        
        return reversed(sorted(differences))

    def eval_action(self, s1, a, values):
        v = 0
        factored_s = self.index_to_factored_state(s1)
        (comp_num, orig_a) = self.action_to_component(a)
        sub_s1 = factored_s[comp_num]
        cur_mdp = self.component_mdps[comp_num]
        for sub_s2 in range(cur_mdp.num_states):
            factored_s[comp_num] = sub_s2
            s2 = self.factored_state_to_index(factored_s)
            v += cur_mdp.transition_dist[orig_a][sub_s1][sub_s2]*values[s2]
        return cur_mdp.rewards[orig_a][sub_s1] + cur_mdp.gamma*v
        

    def optimal_policy(self):
        if not self.value_fn:
            self.compute_value_fn()

        self.opt_policy = [util.argmax(self.actions, 
                                    lambda x: self.eval_action(s, x, self.value_fn))
                                    for s in range(self.num_states)]
        return self.opt_policy


                
