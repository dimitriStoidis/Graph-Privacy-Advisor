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
import pdb
import os
from datetime import datetime
import time


def initiate(hyp_params, train_data, test_data):
    model = GPA(num_class=2, adjacency_matrix=hyp_params.adj_matrix)
    model = model.to('cuda')
    optimizer = getattr(optim, hyp_params.optim)(filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=hyp_params.lr)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'train_set': train_data,
                'test_set': test_data
                }
    return train_model(settings, hyp_params)


def write_log(param_names, dir_):
    log = open(dir_ + "_predslog.txt", 'w')
    log.write('----------------------------------------------------------' + '\n')
    log.write("\nExperiment initiated on :%s\n" + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    log.write("\nModel name: " + param_names.model_name + '\t' + 'lr: ' + str(param_names.lr) + '\n')
    log.write("batch_size " + str(param_names.batch_size) + '\t' + 'epochs ' + str(param_names.num_epochs) + '\n')
    log.write("cardinality " + str(param_names.cardinality) + '\t' 'scene ' + str(param_names.scene))
    log.write("Adjacency Matrix: " + str(param_names.adj_matrix).split('privacy_adjacencyMatrix_')[1] + '\n')
    log.write('----------------------------------------------------------' + '\n')
    return log


# Training and Evaluation

def train_model(settings, hyp_params):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    train_set = settings['train_set']
    test_set = settings['test_set']
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=hyp_params.batch_size, shuffle=True)

    # create .txt file to log results
    checkpoint_dir = './checkpoints/' + hyp_params.model_name
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    log = write_log(hyp_params, checkpoint_dir)

    def train(model_, criterion_, optimizer_, loader_):

        model_.train()
        param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters: {}".format(param_num))

        train_losses = AverageMeter()
        # Initialise classification values for binary, mutliclass and GT labels
        cm_pred,  cm_targets = [], []

        # Initialise Training
        start_epoch_time = time.time()
        for i, (target, full_im, categories, image_name) in enumerate(loader_):
            start_batch_time = time.time()

            batch_size = full_im.shape[0]
            cur_rois_sum = categories[0, 0].clone()

            for b in range(1, batch_size):
                cur_rois_sum += categories[b, 0]

            target = target.cuda(non_blocking=True)  # async=True)

            full_im_var = Variable(full_im).cuda()
            categories_var = Variable(categories).cuda()
            target_var = Variable(target).cuda()

            optimizer_.zero_grad()

            binary_output = model_(full_im_var, categories_var, hyp_params.cardinality, hyp_params.scene)

            loss = criterion_(binary_output, target_var)  # cross-entropy
            train_losses.update(loss.item())

            loss.backward()
            optimizer_.step()

            output_f = F.softmax(binary_output, dim=1)
            output_np = output_f.data.cpu().numpy()

            #  Take GT labels
            targets = target.data.cpu().numpy()
            # Take predictions from Graph model
            preds = [np.argmax(output_np[val]) for val in range(len(output_np))]

            cm_pred = np.append(cm_pred, np.array(preds))
            cm_targets = np.append(cm_targets, targets)

            if i % 20 == 0 and i > 1:
                print("Batch processing time: {:.4f}".format(time.time() - start_batch_time))
                log.write("Batch processing time: {:.4f}".format(time.time() - start_batch_time))
                print("Binary preds: ", preds)
                print("Ground-Truth: ", targets)
                log.write('----------------------------------------------------------' + '\n')
                log.write("\nTraining " + '\n')
                acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1 = get_metrics(cm_targets, cm_pred)
                print("Binary\nUBA(%): {:.4f} Prec (Pub) {:.4f} Rec (Priv){:.4f} Prec (Priv){:.4f}".format(acc, pub_prec, priv_rec, priv_prec))
                log.write('Binary\nUBA(%): {:.4f}'.format(acc) + '\n')
                log.write('Precision (Pub) {:.4f} Recall (Pub){:.4f}'.format(pub_prec, pub_rec) + '\n')
                log.write('Precision (Priv) {:.4f} Recall (Priv){:.4f}'.format(priv_prec, priv_rec) + '\n')

        acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1 = get_metrics(cm_targets, cm_pred)
        print("Binary\nUBA(%): {:.4f} Prec (Pub) {:.4f} Rec (Priv){:.4f} Prec (Priv){:.4f}".format(acc, pub_prec, priv_rec, priv_prec))
        log.write('Binary\nUBA(%): {:.4f}'.format(acc) + '\n')
        log.write('Precision (Pub) {:.4f} Recall (Pub){:.4f}'.format(pub_prec, pub_rec) + '\n')
        log.write('Precision (Priv) {:.4f} Recall (Priv){:.4f}'.format(priv_prec, priv_rec) + '\n')
        log.write(str(cm))

        print("\nEpoch processing time: {:.4f}".format(time.time() - start_epoch_time))
        print("Epoch: ", epoch)
        print("Model:", hyp_params.model_name)
        return

    def evaluate(model_, criterion_, loader_, is_test=False):

        model_.eval()
        cm_pred, cm_targets = [], []

        prediction_scores, target_scores, img_arr = [], [], []

        log.write('\n----------------------------------------------------------' + '\n')
        if is_test:
            log.write("\nTesting " + '\n')
            print("Testing..")
        else:
            log.write("\nValidating" + '\n')
            print("\nValidating..")
        log.write('----------------------------------------------------------' + '\n')

        with torch.no_grad():
            for i, (target, full_im, categories, image_name) in enumerate(loader_):

                # target.shape = [batch_size], full_im.shape = [bs, 3, 448, 448], categories.shape = [bs, 12+1]
                # target label 0 for private and 1 for public
                batch_size = full_im.shape[0]
                cur_rois_sum = categories[0, 0].clone()

                for b in range(1, batch_size):
                    cur_rois_sum += categories[b, 0]
                target = target.cuda(non_blocking=True)  # async=True)

                full_im_var = Variable(full_im).cuda()
                categories_var = Variable(categories).cuda()
                # Input to model

                binary_output = model_(full_im_var, categories_var, hyp_params.cardinality, hyp_params.scene)

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
            log.write(str(cm) + '\n')

            if is_test:
                img_lst = [img for img in img_arr]
                np.savez('./plots/Preds_' + str(hyp_params.model_name) + '.npz', prediction_scores, target_scores,
                         np.array(img_lst))

        return acc, pub_prec, pub_rec, priv_prec, priv_rec, macro_f1

    kfold = KFold(n_splits=hyp_params.K, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_set)):

        print("Fold: ", fold)
        sys.stdout.flush()
        log.write("Fold: " + str(fold) + '\n')
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(train_set, batch_size=hyp_params.batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_set, batch_size=hyp_params.batch_size, sampler=val_sampler)

        best_acc = 0
        best_pub_prec, best_pub_rec = 0, 0
        best_priv_prec, best_priv_rec = 0, 0
        best_macro_f1 = 0
        es = 0

        for epoch in range(1, hyp_params.num_epochs + 1):
            print("Epoch: [", epoch, "]")
            log.write('Epoch: [' + str(epoch) + ']' + "Fold: [" + str(fold) + ']\n')
            sys.stdout.flush()
            train(model, criterion, optimizer, train_loader)
            if epoch % 5 == 0 or epoch == hyp_params.num_epochs + 1:
                val_acc, val_pub_prec, val_pub_rec, val_priv_prec, val_priv_rec, val_macro_f1 = evaluate(model, criterion, val_loader,
                                                                                       is_test=False)
                if val_acc > best_acc:
                    print("\nSaved best UBA(%) model at epoch: " + str(epoch) + '\n')
                    log.write("\nSaved best UBA(%) model at epoch: " + str(epoch) + '\n')
                    save_model(model, name=checkpoint_dir + '/best_acc.pth')
                    best_acc = val_acc
                    es = 0
                if val_macro_f1 > best_macro_f1:
                    log.write("\nSaved best UW-F1 model at epoch: " + str(epoch) + '\n')
                    save_model(model, name=checkpoint_dir + '/best_macro_f1.pth')
                    best_macro_f1 = val_macro_f1
                    es = 0
                if val_pub_prec > best_pub_prec:
                    log.write("Saved best public precision model at epoch: " + str(epoch) + '\n')
                    save_model(model, name=checkpoint_dir + '/best_pub_prec.pth')
                    best_pub_prec = val_pub_prec
                    es = 0
                if val_pub_rec > best_pub_rec:
                    log.write("Saved best public recall model at epoch: " + str(epoch) + '\n')
                    save_model(model, name=checkpoint_dir + '/best_pub_rec.pth')
                    best_pub_rec = val_pub_rec
                    es = 0
                if val_priv_prec > best_priv_prec:
                    log.write("Saved best private precision model at epoch: " + str(epoch) + '\n')
                    save_model(model, name=checkpoint_dir + '/best_priv_prec.pth')
                    best_priv_prec = val_priv_prec
                    es = 0
                if val_priv_rec > best_priv_rec:
                    log.write("Saved best private recall model at epoch: " + str(epoch) + '\n')
                    save_model(model, name=checkpoint_dir + '/best_priv_rec.pth')
                    best_priv_rec = val_priv_rec
                    es = 0
                else:
                    es = es + 1
                    if es >= 10:
                        break
        print('----------------------------------------------------------\n')
        log.write("Testing model with best val UBA(%) .... " + '\n')

        model = load_model(name=checkpoint_dir + '/best_acc.pth')
        evaluate(model, criterion, test_loader, is_test=True)
        try:
            log.write("\nTesting model with best val macro F1 .... " + '\n')
            model = load_model(name=checkpoint_dir + '/best_macro_f1.pth')
            evaluate(model, criterion, test_loader, is_test=True)

            log.write("\nTesting model with best val public precision .... " + '\n')
            model = load_model(name=checkpoint_dir + '/best_pub_prec.pth')
            evaluate(model, criterion, test_loader, is_test=True)

            log.write("\nTesting model with best val public recall .... " + '\n')
            model = load_model(name=checkpoint_dir + '/best_pub_rec.pth')
            evaluate(model, criterion, test_loader, is_test=True)

            log.write("\nTesting model with best val private precision .... " + '\n')
            model = load_model(name=checkpoint_dir + '/best_priv_prec.pth')
            evaluate(model, criterion, test_loader, is_test=True)

            log.write("\nTesting model with best val private recall .... " + '\n')
            model = load_model(name=checkpoint_dir + '/best_priv_rec.pth')
            evaluate(model, criterion, test_loader, is_test=True)
            break
        except:
            print("Unable to load model for testing!")
            pass
        log.write("\nExperiment terminated on :\n" + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

