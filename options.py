import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated arguments
    parser.add_argument('--fed', type=str, default='fedavg', help="federated optimization algorithm")
    parser.add_argument('--rounds', type=int, default=100, help="total number of communication rounds")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=0.3, help="fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="number of local epochs: E")
    # parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--lr_local', type=float, default=0.01, help="client learning rate")
    parser.add_argument('--lr_glob', type=float, default=1, help="server learning rate")

    # other arguments
    parser.add_argument('--dim', type=int, default=1, help='dimension of the primal variable')
    parser.add_argument('--dataset', type=str, default='quadratic', help="name of dataset")
    parser.add_argument('--degree', type=int, default=1, help="degree of polynomial distribution on clients")
    parser.add_argument('--lam', type=float, default=0, help='optional L1 regularization parameter')
    parser.add_argument('--frac_nonzero', type=float, default=1, help='percent sparsity of true value')
    parser.add_argument('--noise', type=float, default=0, help='dimension of the primal variable')
    # parser.add_argument('--model', type=str, default='constant', help='model name')
    # parser.add_argument('--sampling', type=str, default='noniid', help="sampling method")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    # parser.add_argument('--sys_homo', action='store_true', help='no system heterogeneity')
    parser.add_argument('--tsboard', action='store_true', help='tensorboard')
    parser.add_argument('--save_path', type=str, default=None, help='path to save numpy arrays')
    parser.add_argument('--debug', action='store_true', help='debug')
    
    args = parser.parse_args()
    
    return args
