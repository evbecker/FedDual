import autograd.numpy as np
from autograd import grad
from scipy.stats import triang, rv_discrete
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from options import args_parser
from utils import *

def local_update(x_init, local_quad, args):
    grad_quad = grad(local_quad)
    epoch_loss = []
    x = np.array(x_init, dtype=float)
    
    for i in range(args.local_ep):
        epoch_loss.append(local_quad(x))
        x -= args.lr_local*grad_quad(x)
    
    return x, sum(epoch_loss)/len(epoch_loss)

def global_update(x_glob, x_locals, importance, args):
    delta = 0
    for i, x in enumerate(x_locals):
        delta += importance[i]*(x-x_glob)
    x_avg = x_glob + args.lr_glob*delta

    return x_avg



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

    # simulation variable
    x_glob = np.zeros(args.dim)
    
    for iter in trange(args.rounds):
        client_ids, probs = select_clients(m, 'pmf', clients=all_clients, p=p)

        # local updates
        x_locals=[]
        for idx, client_id in enumerate(client_ids):
            client_param = np.array([client_id])
            local_quad = get_local_quad(client_param, args)
            x, loss = local_update(x_glob, local_quad, args)
            x_locals.append(x)
        
        # global update step
        importance = probs/np.sum(probs)
        x_glob = global_update(x_glob, x_locals, importance, args)      
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
        with open(f'{args.save_path}/FedAvg-{args.degree}-{args.frac}-{args.frac_nonzero}-{args.noise}.npy', 'wb') as f:
            np.save(f, np.array(param_L1_hist))
            np.save(f, np.array(param_L2_hist))
            np.save(f, np.array(loss_hist))
            np.save(f, np.array(loss_diff_hist))
    writer.close()
