import argparse
from data_utils import get_data_set
from utils import set_seed
import train_1, test_eval

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Graph Privacy Advisor')
    p.add_argument('--seed', type=int, default=789)
    p.add_argument('--root_dir', type=str, default='')
    p.add_argument('--batch_size', type=int, default=20)
    p.add_argument('--num_class', type=int, default=2)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--K', type=int, default=3, help='number of folds')
    p.add_argument('--optim', type=str, default='Adam',
                   help='optimizer to use (default: Adam)')
    p.add_argument('--model_name', type=str, default='debug')
    p.add_argument('--cardinality', type=str, default='True')
    p.add_argument('--scene', type=str, default='True')
    
    params = p.parse_args()
    set_seed(params.seed)

    params.bbox_dir = params.root_dir + 'pytorch_objectdetecttrack'
    params.adj_matrix = params.root_dir + 'adjacencyMatrix/privacy_adjacencyMatrix_PrivacyAlert_co_occ_binary.npy'
    data_dir = params.root_dir + 'data_preprocess_'

    train_set = get_data_set(params.bbox_dir + '/trainval_PrivacyAlert_bboxes/', partition='trainval')
    test_set = get_data_set(params.bbox_dir + '/test_PrivacyAlert_bboxes/', partition='test')

    print("entering training-validation-testing code\nfor model: ", params.model_name)
    train_1.initiate(params, train_set, test_set)
    # test_eval.initiate(params, test_set)

