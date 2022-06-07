import autograd.numpy as np
from autograd import grad
from scipy.stats import triang, rv_discrete
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from options import args_parser
from utils import *

def soft_threshold(x_init, t):
    # assume for now x is a 1d array
    x = np.copy(x_init)
    if x.shape==():
        x = np.expand_dims(x,0)
    for i in range(len(x)):
        if x[i]>t: 
            x[i]-=t
        elif x[i]>-t:
            x[i]=0
        else:
            x[i]+=t
    return x

def local_update(y_init, local_quad, round, args):
    y = np.array(y_init, dtype=float)
    objective = lambda x : local_quad(x)
    grad_objective = grad(objective)
    epoch_loss = []
    
    for i in range(args.local_ep):
        lr_i = args.lr_local*(args.lr_glob*round*args.local_ep + i)
        x = soft_threshold(y, lr_i*args.lam)
        epoch_loss.append(objective(x))
        y -= args.lr_local*grad_objective(x)
    
    return y, sum(epoch_loss)/len(epoch_loss)

def global_update(y_glob, y_locals, importance, args):
    delta = 0
    for i, y in enumerate(y_locals):
        delta += importance[i]*(y-y_glob)
    y_avg = y_glob + args.lr_glob*delta

    return y_avg


if __name__ == '__main__':
    args = args_parser()
    np.random.seed(args.seed)

    # setting up to save training history
    writer = SummaryWriter()
    loss_hist = []
    loss_diff_hist = []
    param_L1_hist = []
    param_L2_hist = []
    
    # defining "data" distribution across clients 
    m = max(int(args.frac * args.num_clients), 1)   
    all_clients, p = get_client_distribution(args.num_clients, degree=args.degree)
    
    # simulation variables
    y_glob = np.zeros(args.dim)

    for iter in trange(args.rounds):
        client_ids, probs = select_clients(m, 'pmf', clients=all_clients, p=p)

        # local updates of dual variable
        y_locals = []
        for idx, client_id in enumerate(client_ids):
            client_param = np.array([client_id])
            local_quad = get_local_quad(client_param, args)
            y, loss = local_update(y_glob, local_quad, iter, args)
            y_locals.append(y)
        
        # global update step
        importance = probs/np.sum(probs)
        y_glob = global_update(y_glob, y_locals, importance, args)
        x_glob = soft_threshold(y_glob, 2*args.lr_local)
        loss, loss_diff, param_L1, param_L2 = test_quad(x_glob, support=all_clients, p=p, args=args) 

        # tensorboard
        if args.tsboard:
            writer.add_scalar(f"L1 Parameter: Share{args.dataset}, {args.fed}", param_L1, iter)
            writer.add_scalar(f"L2 Parameter: Share{args.dataset}, {args.fed}", param_L2, iter)
            writer.add_scalar(f"Loss: Share{args.dataset}, {args.fed}", loss, iter)
            writer.add_scalar(f"Loss Difference: Share{args.dataset}, {args.fed}", loss_diff, iter)

        # numpy array
        if args.save_path is not None:
            param_L1_hist.append(param_L1)
            param_L2_hist.append(param_L2)
            loss_hist.append(loss)
            loss_diff_hist.append(loss_diff)

    if args.save_path is not None:
        with open(f'{args.save_path}/FedDualAvg-{args.degree}-{args.frac}-{args.frac_nonzero}-{args.noise}.npy', 'wb') as f:
            np.save(f, np.array(param_L1_hist))
            np.save(f, np.array(param_L2_hist))
            np.save(f, np.array(loss_hist))
            np.save(f, np.array(loss_diff_hist))
    writer.close()
