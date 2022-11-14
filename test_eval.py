from networks.gpa import GPA
import torch
from sklearn.model_selection import KFold
from utils import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
from datetime import datetime
import time


def initiate(hyp_params, test_data):
    model = GPA(num_class=2, adjacency_matrix=hyp_params.adj_matrix)
    model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)(filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=hyp_params.lr)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'test_set': test_data
                }
    return train_model(settings, hyp_params)


def write_log(param_names, dir_):
    log = open(dir_ + "_test_img1_predslog.txt", 'w')
    log.write('----------------------------------------------------------' + '\n')
    log.write("\nExperiment initiated on :%s\n" + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    log.write("\nModel name: " + param_names.model_name + '\t' + 'lr: ' + str(param_names.lr) + '\n')
    log.write("batch_size " + str(param_names.batch_size) + '\t')
    log.write("cardinality " + str(param_names.cardinality) + '\t' 'scene ' + str(param_names.scene) + '\t' +
              'object feat ' + str(param_names.object_features) + '\t' + str(param_names.image_features))
    log.write('----------------------------------------------------------' + '\n')
    return log

# Training and Evaluation


def train_model(settings, hyp_params):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    test_set = settings['test_set']
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=hyp_params.batch_size, shuffle=False)

    # create .txt file to log results
    checkpoint_dir = './checkpoints/' + hyp_params.model_name

    log = write_log(hyp_params, checkpoint_dir)

    def evaluate(model_, criterion_, loader_, is_test=False):

        model_.eval()
        test_losses = AverageMeter()
        cm_pred, cm_targets = [], []
        prediction_scores, target_scores, img_arr = [], [], []

        log.write('\n----------------------------------------------------------' + '\n')
        log.write("\nTesting " + '\n')
        print("Testing..")

        with torch.no_grad():
            for i, (target, full_im, bboxes_, categories, image_name) in enumerate(loader_):

                # target label 0 for private and 1 for public
                batch_size = bboxes_.shape[0]
                cur_rois_sum = categories[0, 0].clone()
                bboxes = bboxes_[0, 0:categories[0, 0], :]

                for b in range(1, batch_size):
                    bboxes = torch.cat((bboxes, bboxes_[b, 0:categories[b, 0], :]), 0)
                    cur_rois_sum += categories[b, 0]
                assert (bboxes.size(0) == cur_rois_sum), 'Bboxes num must equal to categories num'

                target = target.cuda(non_blocking=True)  # async=True)

                full_im_var = Variable(full_im).cuda()
                bboxes_var = Variable(bboxes).cuda()
                categories_var = Variable(categories).cuda()
                target_var = Variable(target).cuda()
                # Input to model
                start_batch_time = time.time()
                binary_output = model_(full_im_var, bboxes_var, categories_var,
                                       hyp_params.cardinality, hyp_params.scene,
                                       hyp_params.image_features, hyp_params.object_features)

                print("Batch processing time: {:.4f}".format(time.time() - start_batch_time))
                log.write("\nBatch processing time: {:.4f}\n".format(time.time() - start_batch_time))
                output_f = F.softmax(binary_output, dim=1)

                output_np = output_f.data.cpu().numpy()
                targets = target.data.cpu().numpy()
                preds = np.argmax(output_np, axis=1)
                prediction_scores.append(output_np[:, 0])
                target_scores.append(targets)
                img_arr.append(image_name)

                cm_pred = np.append(cm_pred, np.array(preds))
                cm_targets = np.append(cm_targets, targets)

                print("len: ", i, len(loader_))
                print("Binary preds: ", preds)
                print("Ground-Truth: ", targets)

            acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1 = get_metrics(cm_targets, cm_pred)
            print("Binary\nUBA(%): {:.4f} Precision (Pub) {:.4f} Recall (Priv){:.4f}".format(acc, pub_prec, priv_rec))
            log.write('Binary\nUBA(%): {:.4f}'.format(acc) + '\n')
            log.write('Precision (Pub) {:.4f} Recall (Pub){:.4f}'.format(pub_prec, pub_rec) + '\n')
            log.write('Precision (Priv) {:.4f} Recall (Priv){:.4f}'.format(priv_prec, priv_rec) + '\n')
            log.write('U-F1: {:.4f}'.format(macro_f1) + '\n')
            log.write(str(cm) + '\n')

            if is_test:
                img_lst = [img for img in img_arr]
                np.savez('./plots/Preds_' + str(hyp_params.model_name) + '_test_img1.npz', prediction_scores, target_scores,
                         np.array(img_lst))

        return acc, pub_prec, pub_rec, priv_prec, priv_rec, macro_f1

    # Testing at the end of the Fold
    log.write("\nTesting model with best val macro F1 .... " + '\n')
    model = load_model(name=checkpoint_dir + '/best_macro_f1.pth')
    evaluate(model, criterion, test_loader, is_test=True)
    log.write("\nExperiment terminated on :\n" + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))