from __future__ import division

import mdp_base as mdp
reload(mdp)
import numpy as np
import scipy.io
import util as u
reload(u)
import cProfile, random, pdb
import search
import IPython as ipy
from pdb import pm
import sys, time
import argparse, h5py, socket

def r_and_d_mdp(R_success, safe_wait_p, risky_wait_p, risky_success_p):
    """
    States: state wait_safe wait_risky produce_safe produce_risky_success produce_risky_fail
    """
    T = {}
    R = {}
    num_states = 6
    actions = ['safe', 'risky']
    rewards = np.array([0, 0, 0, R_success, R_success, 0])
    t_safe = np.array([[0, 1, 0, 0, 0, 0],
                       [0, safe_wait_p, 0, 1 - safe_wait_p, 0, 0],
                       [0, 0, risky_wait_p, 0, risky_success_p, 1-risky_wait_p-risky_success_p],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
    t_risky = t_safe.copy()
    t_risky[0, 1] = 0
    t_risky[0, 2] = 1
    T['safe'] = t_safe
    T['risky'] = t_risky
    R['safe'] = rewards
    R['risky'] = rewards
    # print T['safe']
    # print T['risky']
    # print R
    return mdp.MDP(actions, num_states, T, R, 0.9)

def multiple_r_and_d_mdp(seeds, gamma):
    try:
        N = len(seeds)
    except TypeError:
        N = seeds
        seeds = None
    params = []
    for i in range(N):
        if seeds:
            random.seed(seeds[i])
        p = {}
        p['R_success'] = random.random()
        p['safe_wait_p'] = 0.75
        p['risky_wait_p'] = 0.5
        # p['risky_success_p'] = (np.random.rand() / 2 + 0.5) * (1 - p['risky_wait_p'])
        p['risky_success_p'] = (0.8) * (1 - p['risky_wait_p'])
        params.append(p)
    c_mdps = [r_and_d_mdp(**p) for p in params]
    return mdp.MultipleMDP(c_mdps, gamma)

def deterministic_chain_mdp(num_states=3, rewards=None):
    if not rewards:
        rewards = np.linspace(10, 3, num_states)
    T = {}
    T_a = np.zeros((num_states, num_states))
    for i in range(num_states-1):
        T_a[i, i+1] = 1
    T_a[num_states-1, num_states-1] = 1
    R = {}
    R['a1'] = rewards
    T['a1'] = T_a
    return mdp.MDP(['a1'], num_states, T, R, .9)

def multiple_grid_mdp(seeds, n_gold_locs, gamma, dim):
    params = []
    for s in seeds:
        params.append(gen_multi_reward_params(seed = s, num_gold_locs = n_gold_locs,
                                              x_max=dim, y_max=dim, gamma=gamma))
    c_mdps = [multi_reward_grid_mdp(**p) for p in params]
    return mdp.MultipleMDP(c_mdps, gamma)



x_max_default = 10
y_max_default = 10

def gen_multi_reward_params(x_max = x_max_default, y_max = y_max_default, 
                            max_gold_r=1, max_t_r=.02, num_gold_locs=3, 
                            num_reset_locs=5, seed = 0, gamma=.9):
    # if seed:
    #     print 'seeding'
    #     random.seed(seed)
    num_states = (x_max+1)*(y_max+1)#zero indexed grid
    permuted_states = range(num_states)
    random.shuffle(permuted_states)
    reset_p = {}
    reward_locs = {}
    for r in permuted_states[:num_gold_locs]: 
        reward_locs[r] = max_gold_r*random.random()
        reset_p[r] = np.random.randint(num_states)
    for r in permuted_states[num_gold_locs:3*num_gold_locs]: 
        reward_locs[r] = max_gold_r*random.random() / 5.0
        # reset_p[r] = np.random.randint(num_states)

    # random.shuffle(permuted_states)
    # reset_p = np.zeros(num_states)
    # for s in permuted_states[:num_reset_locs]:
    #     reset_p[s] = 1/float(num_reset_locs)
    actions = ['U', 'D', 'L', 'R']
    t_rewards = {}
    for s in range(num_states):
        for a in actions:
            t_rewards[(s, a)] = random.random() * max_t_r
            
    success_p = .8
    params = {'reward_locs':reward_locs,
              'reset_p': reset_p,
              'transition_rewards':t_rewards,
              'success_p':success_p,
              'x_max':x_max,
              'y_max':y_max,
              'gamma':gamma }
    # print params['reward_locs'], params['reset_p']
    return params



def multi_reward_grid_mdp(reward_locs, reset_p, transition_rewards, success_p,
                          x_max = x_max_default, y_max = y_max_default,  gamma = .9):
    num_states = (x_max+1)*(y_max+1)#zero indexed grid
    actions = {'U':[0, 1], 'D':[0, -1], 'L':[-1, 0], 'R':[1, 0]}
    s_index = lambda (x,y): x + (x_max+1)*y
    T = {}
    for a in actions:
        T_a = np.zeros((num_states, num_states))
        for x in range(x_max+1):
            for y in range(y_max+1):
                cur_s = (x, y)
                cur_s_i = s_index(cur_s)
                if cur_s_i in reset_p:
                    T_a[cur_s_i,reset_p[cur_s_i]] = 1 
                    continue
                for rand_a in actions:##if the action fails we randomly stay in the same state or 
                ##move to an adjacent state
                    next_s = u.componentAdd(cur_s, actions[rand_a])
                    next_s = (u.clip(next_s[0], 0, x_max), u.clip(next_s[1], 0, y_max))
                    next_s_i = s_index(next_s)
                    T_a[cur_s_i, next_s_i] += (1-success_p)/5.0
                T_a[cur_s_i, cur_s_i] += (1-success_p)/5.0
                next_s = u.componentAdd(cur_s, actions[a])
                next_s = (u.clip(next_s[0], 0, x_max), u.clip(next_s[1], 0, y_max))
                next_s_i = s_index(next_s)
                T_a[cur_s_i, next_s_i] += success_p
    
        T[a] = T_a

    R = {}
    for a in actions:
        R[a] = np.zeros(num_states)
        for s in range(num_states):
            R[a][s] = 0#transition_rewards[(s, a)]
        for r_loc in reward_locs:
            R[a][r_loc] += reward_locs[r_loc]

    act_list = T.keys()

    m = mdp.MDP(act_list, num_states, T, R, gamma)
    return m

def gen_bernoulli_mdp_params(sample_bounds, prior_weight, research_cost, gamma = 0.9, seed = 0):
    if seed:
        random.seed(seed)
    prior_fails = random.random() * (prior_weight/float(3)) + prior_weight/float(3)
    prior = (prior_weight - prior_fails, prior_fails)
    r_cost = random.random() * (research_cost[1] - research_cost[0]) + research_cost[0]
    sample_max = random.choice(range(sample_bounds[0], sample_bounds[1]))
    params = {'sample_max': sample_max,
              'prior_dist': prior,
              'improve_cost': r_cost,
              'gamma': gamma}
    return params

max_size = 50000

def sparse_bernoulli_mrp(sample_max=50, exp_sparsity = .9, participate_cost = .1, prior_dist=(1, 1), gamma=0.9):
    """
    Sparse Bernoulli MRP:
        States
        ------
        (a, b, I)
        Number of successes/failures observed (beta-bernoulli state space)
        I is an indicator for if the next game is important

        Actions
        -------
        'act': 
        
        If a + b = sample_max, then we select this drug for long term deployment. Observe
        long term reward E[V(\theta) | a, b]
    """
    alpha, beta = prior_dist
    state_info = {'num_states': 0}
    def s_index(s):
        if s not in state_info:
            state_info[s] = state_info['num_states']
            state_info['num_states'] += 1
        return state_info[s]
    s_ret = s_index('retire')
    T = {'act': np.zeros((max_size, max_size)), 'retire': np.zeros((max_size, max_size))}
    T['retire'][:, s_ret] = 1
    T['act'][s_ret, s_ret] = 1
    R = {'act': np.zeros(max_size), 'retire':np.zeros(max_size)}
    for n_fails in range(sample_max):
        for n_success in range(sample_max - n_fails):
            a = n_success + alpha
            b = n_fails + beta
            success_p = a / float(a + b)
            for I in range(2):                
                s_ind = s_index((a, b, I))
                if n_success + n_fails == sample_max - 1:
                    # final states just let us sample from them forever
                    s_plus_0, s_minus_0 = s_index((a, b, 0)), s_index((a, b, 0))
                    s_plus_1, s_minus_1 = s_index((a, b, 1)), s_index((a, b, 1))
                else:                    
                    # we update beliefs
                    s_plus_0, s_minus_0 = s_index((a+1, b, 0)), s_index((a, b+1, 0))
                    s_plus_1, s_minus_1 = s_index((a+1, b, 1)), s_index((a, b+1, 1))
                # we don't have an important game next with probability s
                T['act'][s_ind, s_plus_0] += success_p * exp_sparsity
                T['act'][s_ind, s_minus_0] += (1-success_p) * exp_sparsity
                # we do haven an important game next with prob 1-s
                T['act'][s_ind, s_plus_1] += success_p * (1-exp_sparsity)
                T['act'][s_ind, s_minus_1] += (1-success_p) * (1-exp_sparsity)
                # normalize
                T['act'][s_ind, :] /= np.sum(T['act'][s_ind, :])
                # we always pay particpation cost
                R['act'][s_ind] = -participate_cost
                # if its an important game, we collect the expected value of the game
                # equivalent to getting value 1 and -1 from a decision standpoint
                if I: 
                    R['act'][s_ind] += (1)*success_p + (-1)*(1-success_p)
    num_states = state_info['num_states']
    T['act'] = T['act'][:num_states, :num_states]
    T['retire'] = T['retire'][:num_states, :num_states]
    R['act'] = R['act'][:num_states]
    R['retire'] = R['retire'][:num_states]
    # transform gamma 
    # this ensures that the (expected) rate at which important
    # games arive remains constant
    def gamma_mod(gamma_hat, s):
        return  1 - (1-gamma_hat)*(1-s)
    gamma = gamma_mod(gamma, exp_sparsity)
    m = mdp.MDP(['act', 'retire'], num_states, T, R, gamma)
    # return the mdp and retirement reward to evaluate at
    return m, state_info

def multiple_bernoulli_mdp(seeds, sample_bounds=(8, 12), prior_weight=5, research_cost=(0, 3), gamma=0.9):
    params = []
    for s in seeds:
        params.append(gen_bernoulli_mdp_params(sample_bounds, prior_weight, 
                                               research_cost, seed=s, gamma=gamma))
    c_mdps = [bernoulli_design_mdp(**p) for p in params]
    print "mdp params:"
    for p in params:
        print p
        raw_input("GO?")
    return mdp.MultipleMDP(c_mdps, gamma)

improve_dist = {3: 0.0219, 
               2: 0.2285, 
               1: 0.4991, 
               0: 0.2285,
               -1: 0.0219}
    
def bernoulli_design_mdp(sample_max=50, prior_dist=(3, 3), improve_cost=1, gamma=0.9):
    """
    Drug Design MDP:
        States
        ------
        (a, b)
        Number of successes/failures observed (beta-bernoulli state space)

        Actions
        -------
        Test: check the current drug [transitions to (a+1, b)/(a, b+1) -- reward R for saving someone
                                      -R for losing someone]
        Improve: improve the quality of the current drug [transitions to (a+1, b)/(a+2, b)...etc], occurs
                 at a fixed constant cost
                     0.0219     0.2285    0.4991   0.2285   0.0219
                 (a+3, b-3)  (a+2, b-2) (a+1, b-1) (a, b) (a-1, b-1) 
        
        If a + b = sample_max, then we select this drug for long term deployment. Observe
        long term reward E[V(\theta) | a, b]
    """
    alpha, beta = prior_dist
    num_states = int((sample_max)*(sample_max+1)/float(2))
    s_index = lambda (a, b): (a + b - beta - alpha)*(a + b-beta-alpha + 1)/ float(2) + a - alpha
    T = {'test': np.zeros((num_states, num_states)), 'improve': np.zeros((num_states, num_states))}
    R = {'test': np.zeros(num_states), 'improve': -1 * improve_cost * np.ones(num_states)}
    for n_fails in range(sample_max):
        for n_success in range(sample_max - n_fails):
            a = n_success + alpha
            b = n_fails + beta
            success_p = a / float(a + b)
            s_ind = s_index((a, b))
            if n_fails + n_success < sample_max - 1:
                ## we get an update
                for improve_amount, improve_p in improve_dist.iteritems():
                    if a + improve_amount < sample_max + alpha and b-improve_amount > beta\
                            and a + improve_amount > alpha and b - improve_amount < sample_max + beta:
                        s_new = s_index((a + improve_amount, b-improve_amount))
                        T['improve'][s_ind, s_new] += improve_p                    
                    else:
                        T['improve'][s_ind, s_ind] += improve_p
                T['improve'][s_ind] *= (1/np.sum(T['improve'][s_ind]))
                s_plus, s_minus = s_index((a+1, b)), s_index((a, b+1))
            else:
                s_plus, s_minus = s_ind, s_ind
                T['improve'][s_ind, s_ind] = 1
            T['test'][s_ind, s_plus] += success_p
            T['test'][s_ind, s_minus] += 1-success_p
            T['test'][s_ind] *= (1/np.sum(T['test'][s_ind]))
            R['test'][s_ind] = (1)*success_p + (-1)*(1-success_p)
            
            
    min_r = min(np.min(R['test']), np.min(R['improve']))
    R['test'] -= min_r
    R['improve'] -= min_r
    # Normalize transition matrices
    T['test'] = T['test'] / T['test'].sum(axis=1)
    T['improve'] = T['improve'] / T['improve'].sum(axis=1)
    m = mdp.MDP(['test', 'improve'], num_states, T, R, gamma)
    return m

def grid_mdp(x_max=3, y_max=4, success_p=.6):
####################Create an example MDP transition model -- grid navigation####################
####20 States
####4 actions
#      |     | |    | |
#      |     | |    | |
#      |     | |    | |
#      |     | |-20 | |
#      |(0,0)| |+50 | |
#
#
#
#
#
    num_states = (x_max+1)*(y_max+1) + 1 #zero indexed grid + sink state

    actions = {'U':[0, 1], 'D':[0, -1], 'L':[-1, 0], 'R':[1, 0]}
    s_index = lambda (x,y): x + (x_max+1)*y
    T = {}
    for a in actions:
        T_a = np.zeros((num_states, num_states))
        T_a[num_states-1, num_states-1] = 1
        for x in range(x_max+1):
            for y in range(y_max+1):
                cur_s = (x, y)
                cur_s_i = s_index(cur_s)
                if cur_s in [(0, x_max), (1, x_max-1)]:
                    T_a[cur_s_i, num_states-1] = 1
                    continue
                for rand_a in actions:##if the action fails we randomly stay in the same state or 
                ##more to an adjacent state
                    next_s = u.componentAdd(cur_s, actions[rand_a])
                    next_s = (u.clip(next_s[0], 0, x_max), u.clip(next_s[1], 0, y_max))
                    next_s_i = s_index(next_s)
                    T_a[cur_s_i, next_s_i] += (1-success_p)/5.0
                T_a[cur_s_i, cur_s_i] += (1-success_p)/5.0
                next_s = u.componentAdd(cur_s, actions[a])
                next_s = (u.clip(next_s[0], 0, x_max), u.clip(next_s[1], 0, y_max))
                next_s_i = s_index(next_s)
                T_a[cur_s_i, next_s_i] += success_p
    
    
        T[a] = T_a

    R = {}
    for a in actions:
        R[a] = 20*np.ones(num_states)
        R[a][s_index((x_max, 0))] = 70
        R[a][s_index((x_max-1, 1))] = 0

    act_list = T.keys()

    gamma = .9
    m = mdp.MDP(act_list, num_states, T, R, gamma)
    return m




digit_to_symb = dict(zip([str(i) for i in range(10)], [str(unichr(97+i)) for i in range(10)]))
symb_to_digit = dict([(x, y) for y, x in digit_to_symb.items()])

def int_to_symbol(i):
    s = str(i)
    result = ''
    for c in s:
        result += digit_to_symb[c]
    return result

def write_spudd_model(comp_mdps, outfname):
    """
    we just assuming the default params from the multi_rewards_grid_mdp

    Variables: x_i y_i
    Actions: u_i, d_i, r_i, l_i
    """
    N = len(comp_mdps)
    var_names = ['s_{}'.format(i) for i in range(N)]
    var_str = '(variables'
    for i in range(N):
        var_str += '\n(s_{} '.format(i)
        for l in range(comp_mdps[i].num_states):
            var_str += '{} '.format(int_to_symbol(l))
        var_str += ')'
    var_str += ')\n'
    ##for each action, build up dict with dd for transition dist
    def stay_still_dd(i):
        ret_str = 'dd default{}\n'.format(i)
        v = 's_{}'.format(i)
        ret_str += "({} ".format(v)
        for l in range(comp_mdps[i].num_states):
            s_l = int_to_symbol(l)
            ret_str += '({} ({}\' ({} (1.0)) '.format(s_l, v, s_l)
            for j in range(comp_mdps[i].num_states):
                if j != l:
                    ret_str += "({} (0.0)) ".format(int_to_symbol(j))
            ret_str += '))'
        ret_str += ")\nenddd\n"
        return ret_str
    def move_transition_dist(a, i):
        ret_str = 'action {}_{}\n'.format(a, i)
        for j in range(N):
            if j != i:
                ret_str += 's_{}  (SAMEs_{})\n'.format(j, j)
        cur_mdp = comp_mdps[i]        
        T_a = cur_mdp.transition_dist[a]
        R_a = cur_mdp.rewards[a]
        ret_str += 's_{}  (s_{} \n' .format(i, i)
        for s1 in range(cur_mdp.num_states):
            s1_symb = int_to_symbol(s1)
            ret_str += '({}  ('.format(s1_symb)
            for s2 in range(cur_mdp.num_states):
                ret_str += '{} '.format(T_a[s1, s2])
            ret_str += '))\n'
        ret_str += ')\n'
        ret_str += 'cost  (s_{}  '.format(i)
        for s1 in range(cur_mdp.num_states):
            ret_str += '\n({} (-{})) '.format(int_to_symbol(s1), R_a[s1])
        ret_str += ')\nendaction\n'
        return ret_str
    out_str = ''
    # write variables
    out_str += var_str
    # write actions
    for i in range(N):
        for a in comp_mdps[i].actions:
            out_str += move_transition_dist(a, i)
    # write discount/actions
    out_str += 'reward (0.0)\ndiscount {}\ntolerance {}'.format(comp_mdps[0].gamma, .01)
    with open(outfname, 'w') as outf:
        outf.write(out_str)    

