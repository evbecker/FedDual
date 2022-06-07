import autograd.numpy as np
from autograd import grad
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from options import args_parser
from utils import *

def approx_local_update(x_init, y, p, local_quad, args):
    y = np.array(y, dtype=float)
    objective = lambda x : local_quad(x) - np.dot(y/p,x)
    grad_objective = grad(objective)
    epoch_loss = []
    x = np.array(x_init, dtype=float)
    
    for i in range(args.local_ep):
        epoch_loss.append(objective(x))
        x -= args.lr_local*grad_objective(x)
    
    return x, sum(epoch_loss)/len(epoch_loss)

def exact_local_update(client_param, y, p, local_quad):
    y = np.array(y, dtype=float)
    objective = lambda x : local_quad(x) - np.dot(y/p,x)
    x = (1+y)/client_param
    
    return x, objective(x)

def global_update(x_locals, client_ids, importance, args):
    xhat = {}
    coeffs = np.array([1/param for param in client_ids])
    x_arr = np.array([x_locals[id] for id in client_ids])
    x_sum = np.dot(coeffs, x_arr)

    for idx, client_id in enumerate(client_ids):
        xhat[client_id] = coeffs[idx]*x_arr[idx] - (coeffs[idx]/np.sum(coeffs))*x_sum

    return xhat


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
    x_locals = {id : np.zeros(args.dim) for id in all_clients}
    y_locals = {id : np.zeros(args.dim) for id in all_clients}
     
    for iter in trange(args.rounds):
        client_ids, probs = select_clients(m, 'pmf', clients=all_clients, p=p)

        # local updates of primal variable
        for idx, client_id in enumerate(client_ids):
            client_param = np.array([client_id])
            local_quad = get_local_quad(client_param, args)
            x, loss = approx_local_update(x_locals[client_id], y_locals[client_id], probs[idx], 
                                          local_quad, args)
            x_locals[client_id] = x
        
        # global update step
        importance = probs/np.sum(probs)
        xhat = global_update(x_locals, client_ids, importance, args)
        test_id, _ = select_clients(1, 'pmf', clients=all_clients, p=p)      
        loss, loss_diff, param_L1, param_L2 = test_quad(x_locals[test_id[0]], support=all_clients, p=p, args=args) 

        # local updates of dual variable
        for idx, client_id in enumerate(client_ids):
            y_locals[client_id] -= args.lr_local*xhat[client_id]

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
        with open(f'{args.save_path}/FedDCD-{args.degree}-{args.frac}-{args.frac_nonzero}-{args.noise}.npy', 'wb') as f:
            np.save(f, np.array(param_L1_hist))
            np.save(f, np.array(param_L2_hist))
            np.save(f, np.array(loss_hist))
            np.save(f, np.array(loss_diff_hist))
    writer.close()
