import autograd.numpy as np
from scipy.stats import triang, rv_discrete

# returns 1/2*||A^(1/2)[x-c]||_2^2
def get_local_quad(client_param, args):
    d = args.dim
    A = np.zeros((d,d))
    np.fill_diagonal(A,client_param)
    sqrtA = np.sqrt(A)
    c = np.ones((d,))/client_param
    c[int(d*args.frac_nonzero):] = 0
    c += np.random.normal(scale=args.noise, size=(d,))
    return lambda x: 0.5*np.linalg.norm(sqrtA@(x-c))**2

def test_quad(x_glob, support, p, args):
    frac_nonzero = args.frac_nonzero
    d = args.dim
    distr = rv_discrete(values=(support,p))
    # true param and function val
    client_param = lambda z : z
    true_a = distr.expect(client_param)
    true_A = np.zeros((d,d))
    np.fill_diagonal(true_A, true_a)
    mask = np.ones((d,))
    mask[int(d*frac_nonzero):] = 0
    true_param = np.linalg.pinv(true_A)@mask
    true_local = lambda z: 0.5*np.dot(true_param,true_param)*z - np.dot(true_param,mask) + 0.5*int(d*frac_nonzero)/z
    true_global = distr.expect(true_local)
    # estimated parameter and function vals
    est_param = x_glob
    est_local = lambda z: 0.5*np.dot(est_param,est_param)*z - np.dot(est_param,mask) + 0.5*int(d*frac_nonzero)/z
    est_global = distr.expect(est_local)

    loss_diff = est_global-true_global
    test_loss = est_global
    param_L2 = np.linalg.norm(true_param-est_param)
    param_L1 = np.linalg.norm(true_param-est_param, ord=1)
    if args.debug:
        print(f'True param: {true_param}; True global function val: {true_global}')
        print(f'Est. param: {est_param}; Est. global function val: {est_global}')

    return test_loss, loss_diff, param_L1, param_L2

def get_client_distribution(num, min=0.5, max=2, degree=1):
	clients = np.linspace(min, max, num)
	p = clients**degree/np.sum(clients**degree)
	return clients, p

def select_clients(m, distr_type='triangular', **kwargs):
    if distr_type=='triangular':
        client_ids = triang.rvs(c=kwargs['mode'],loc=kwargs['left'],scale=kwargs['right'], size=m)
        probs = triang.pdf(x=idxs_clients, c=kwargs['mode'],loc=kwargs['left'],scale=kwargs['right'])
    elif distr_type=='uniform':
        select_idx = np.random.choice(range(len(kwargs['clients'])), m, replace=False)
        client_ids = kwargs['clients'][select_idx] 
        probs = [1./len(kwargs['clients']) for idx in select_idx]
    elif distr_type=='pmf':
        select_idx = np.random.choice(range(len(kwargs['clients'])), m, replace=False, p=kwargs['p'])
        client_ids = kwargs['clients'][select_idx]
        probs = kwargs['p'][select_idx]
    else:
        print(f'no distrbution defined for {distr_type}')
    return client_ids, probs