import argparse

def arg_parse():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UMKL-G', help='UMKL-G, UMKL, sparseUMKL')
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Name of the dataset.')
    parser.add_argument('--num', type=int, default=1, help='No. of the experiment.')
    
    parser.add_argument('--loss_fun', type=str, default='PKL', help='Loss function: `PKL`(power KL Divergence), `MKL`(multi KL Divergence), `PCE`(power Cross Entropy), `MCE`(multi Cross Entropy)')
    parser.add_argument('--power', type=int, default=2, help='Hyperparameter in some loss funcitons.')
    parser.add_argument('--set_fixed_ground_truth', type=bool, default=False, help='a fixed or dynamic target distribution Q.')
    parser.add_argument('--forward_loss', type=bool, default=False, help='Define a forward KL or reverse KL.')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Size of neighborbood (Only for sparse-UMKL)')

    parser.add_argument('--init_type', type=str, default='uniform', help='Three method to initialize weights: `uniform`, `eigen`, or `eigen_inv`')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--pre_epochs', type=int, default=200,
                        help='Number of epochs to train the GCN.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--perturb', type=bool, default=False, help='Add noise to the kernel matrix.')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Standard deviation of the noise.')

    parser.add_argument('--test', type=bool, default=False, help='Test mode.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size.')

    args = parser.parse_args()
    return args