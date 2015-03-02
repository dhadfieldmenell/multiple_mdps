from mdp_gen import gen_multi_reward_params, multi_reward_grid_mdp, multiple_grid_mdp, write_spudd_model, multiple_bernoulli_mdp
from mdp_base import MultipleMDP
from factored_alp import factored_alp, eval_approx_qf
import search
reload(search)
from search import local_search_vi
import h5py, random, time
from sys import stdout
from pdb import pm
import argparse
import IPython as ipy
import time
import numpy as np
import util as u
import subprocess
n_mdp_vals = [2, 6, 10, 14]
n_gold_loc_vals = [1, 3, 5, 7]

SPUDD_LOC = '/home/dhm/src/spudd-3.6.2/Spudd/bin/linux/Spudd'
SPUDD_FNAME = 'test.mdp'
SPUDD_OUTF = 'data/UAI/comparison_timing.h5'

def compare(outf = SPUDD_OUTF, n_trials=30, mdp_type='grid'):
    n_mdps = [2, 3, 4, 5, 6]#, 7, 8, 9, 10, 11, 12, 13, 14]
    if mdp_type=='grid':
        params = {'n_gold_locs': 3,
                  'gamma': 0.9, 
                  'dim': 4}
        gen_mdp = multiple_grid_mdp
    elif mdp_type == 'bernoulli':
        params={'sample_bounds': (8, 11),
                'prior_weight': 5,
                'research_cost': (1, 3),
                'gamma': 0.9}
        gen_mdp = multiple_bernoulli_mdp
    outf = h5py.File(outf, 'a')
    if 'mdp_type' in outf:
        assert str(outf['mdp_type'][()]) == mdp_type
    else:
        outf['mdp_type'] = mdp_type
    do_spudd = True
    do_singh_cohn = True
    for n in n_mdps:
        if str(n) in outf:
            del outf[str(n)]
        result_g = outf.create_group(str(n))
        result_g['spudd_times']        = -1 * np.ones(n_trials)
        result_g['singh_cohn_times']   = -1 * np.ones(n_trials)
        result_g['blsvi_times']        = -1 * np.ones(n_trials)
        result_g['num_expands']        = -1 * np.ones(n_trials)
        result_g['singh_cohn_expands'] = -1 * np.ones(n_trials)
        result_g['seeds']              = np.zeros((n_trials, n))
        result_g['states']             = np.zeros((n_trials, n))
        result_g['size']               = gen_mdp(
            [random.random() for _ in range(n)], **params).num_states
        for i in range(n_trials):
            mdp_seeds = [random.random() for _ in range(n)]
            result_g['seeds'][i, :] = mdp_seeds
            m = gen_mdp(mdp_seeds, **params)
            start_state = random.randint(0, m.num_states)
            start_state=0
            start_state = u.tuplify(m.index_to_factored_state(start_state))
            result_g['states'][i] = start_state
            start = time.time()
            # try:                            
            wi_frac = m.frontier_alg()
            m.build_sparse_transitions()
            m.compute_lb_mdp()
            search_a, num_expansions, sub_opt = local_search_vi(m, start_state, semi_verbose = True, max_expand=10000)
            # except Exception as e:
            #     print e
            #     print 'Error encountered in search!'
            #     sub_opt = 10
            #     num_expansions = 1000000
            time_taken = time.time() - start
            result_g['num_expands'][i] = num_expansions
            if sub_opt <= 10**-8:
                result_g['blsvi_times'][i] = time_taken
                print 'BBVI {}'.format(time_taken)
            else:
                print "BBVI DID NOT CONVERGE"

            # print n, num_expansions
            if do_spudd:
                write_spudd_model(m.component_mdps, SPUDD_FNAME)
                start = time.time()
                res = subprocess.Popen([SPUDD_LOC, SPUDD_FNAME]).wait()
                time_taken = time.time() - start
                if res >= 0:
                    result_g['spudd_times'][i] = time_taken
                    print 'SPUDD {}'.format(time_taken)
                else:
                    print 'SPUDD DID NOT CONVERGE'
            else:
                print 'skipping SPUDD'
            outf.flush()
            if do_singh_cohn:
                start = time.time()
                try:
                    search_a, num_expansions, sub_opt = local_search_vi(m, start_state, semi_verbose = True, 
                                                                        max_expand=10000, use_wi = False, 
                                                                        min_hval_decrease_rate = .1)             
                except Exception as e:
                    print e
                    print 'Error encountered in search!'
                    sub_opt = 10
                    num_expansions = 1000000
                time_taken = time.time() - start
                result_g['singh_cohn_expands'][i] = num_expansions
                if sub_opt <= 10**-8:
                    result_g['singh_cohn_times'][i] = time_taken
                    print 'SINGH_COHN {}'.format(time_taken)
                else:
                    print "SINGH_COHN DID NOT CONVERGE"
        do_spudd = not np.all(result_g['spudd_times'][:] == -1)
        do_singh_cohn = not np.all(result_g['singh_cohn_times'][:] == -1)

            
DEFAULT_MDP_PARAMS = {'n_gold_locs' : 5,
                      'gamma' : .9,
                      'dim' : 20}

DEFAULT_NUM_ARMS = 10

def do_single_run(num_trials, mdp = None, verbose=False):
    if mdp == None:
        mdp = multiple_grid_mdp([random.random() for i in range(DEFAULT_NUM_ARMS)],
                                **DEFAULT_MDP_PARAMS)
    # mdp.build_sparse_transitions()
    mdp.frontier_alg()
    mdp.compute_lb_mdp()
    total_expansions = []
    start_states = [mdp.get_rand_state(wc_violations_only=False) for i in range(num_trials)]
    print "testing with {:4} states".format(float(mdp.num_states))
    for i, s in enumerate(start_states):
        print s
        start_state = u.tuplify(s)
        # try:
        search_a, num_expansions, sub_opt = local_search_vi(mdp, start_state, 
                                                            semi_verbose=verbose,
                                                            max_expand=100000)
        total_expansions.append(num_expansions)
        # except Exception, e:
        #     print 'Search Failed', e
        #     total_expansions.append(-1)
    print total_expansions
    return total_expansions

def proc_data_file(fname):
    f = h5py.File(fname, 'r')
    time_res = dict([(c, []) for c in exp_conditions[:-1]])
    for c in exp_conditions[:-1]:
        for t in range(10):
            for r in range(40):
                time_res[c].append(f[str(c)][str(t)][str(r)]['total_reward'][()])
    for c in time_res:
        time_res[c] = np.mean(time_res[c])
    return time_res

def proc_all_experiment_files():
    results = {}
    for n_mdp in n_mdp_vals:
        for n_gold_locs in n_gold_loc_vals:
            fname = 'data/{}-mdps-{}-gold-locs.h5'.format(n_mdp, n_gold_locs)
            try:
                results[(n_mdp, n_gold_locs)] = proc_data_file(fname)
            except:
                continue
    ipy.embed()




if __name__ == '__main__':
    import sys; sys.setrecursionlimit(100000)
    parser = argparse.ArgumentParser()
    parser.add_argument('mdp_type', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--n_trials', type=int, default=10)

    # parser.add_argument('exp_file', type=str)
    # parser.add_argument('condition', type=str)
    
    args = parser.parse_args()
    compare(n_trials=args.n_trials, mdp_type=args.mdp_type)
    # num_exp = do_single_run(args.n_trials, verbose = args.verbose)
    # print 'num expansions:'
    # print num_exp
