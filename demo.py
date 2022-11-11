from networks.gpa_demo import GPA_demo
import torch
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
import argparse
from data_utils import get_data_set
from utils import set_seed
import test_eval
from PIL import Image
import torchvision.transforms as transforms
from yolov3_objecte_detection import detect_image


p = argparse.ArgumentParser(description='Graph Privacy Advisor')
p.add_argument('--seed', type=int, default=789)
p.add_argument('--root_dir', type=str, default='')
p.add_argument('--image_name', type=str, default='xyz.jpg')
p.add_argument('--model_name', type=str, default='gpa')
p.add_argument('--cardinality', type=bool, default=True)
p.add_argument('--scene', type=bool, default=True)
p.add_argument('--image_size', type=int, default=416)  # yolov3 hyper-parameter
p.add_argument('--rois', type=int, default=12)

params = p.parse_args()
set_seed(params.seed)

full_transform = transforms.Compose([
    transforms.Resize((448, 448)),  # resize to (3, 448, 448)
    transforms.ToTensor()])
root_dir = ''

class_path = root_dir + '/data_preprocess_/config/coco.names'

classes = load_classes(class_path)


def get_image(image_name):

    img_pil = Image.open(params.root_dir + image_name).convert('RGB')

    detections = detect_image(img_pil)
    img_size = params.image_size
    img = np.array(img_pil)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    categs, obj_categ, bboxes = [], [], []
    print("\nDetected object categories:")
    if detections is not None:
        # browse detections
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            bb = [x1.item(), y1.item(), box_w.item(), box_h.item()]
            bboxes.append(bb)
            categs.append(int(cls_pred))
            obj_categ.append(classes[int(cls_pred)])
    else:
        categs.append(80)
        obj_categ.append('background')
    print(obj_categ)

    max_rois_num = params.rois
    categories = torch.IntTensor(max_rois_num + 1).fill_(-1)
    categories[0] = len(categs)

    if categories[0] > max_rois_num:
        categories[0] = max_rois_num
    else:
        categories[0] = categories[0]
    end_idx = categories[0] + 1
    categories[1: end_idx] = torch.IntTensor(categs)[
                             0:categories[0]]  # e.g; [ 5,  0,  0,  0,  7, 72, -1, -1, -1, -1, -1, -1, -1]
    return img_pil, categories.unsqueeze(0), image_name


def evaluate(model_name, img_name):
    model = load_model(name=model_name + '/best_macro_f1.pth')
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        full_im, categories, image_name = get_image(img_name)
        full_im = full_transform(full_im)
        full_im_var = Variable(full_im).cuda()
        categories_var = Variable(categories).cuda()
        # Input to model
        start_batch_time = time.time()
        binary_output = model(full_im_var, categories_var,
                               params.cardinality, params.scene)
        output_f = F.softmax(binary_output, dim=1)
        output_np = output_f.data.cpu().numpy()
        preds = np.argmax(output_np, axis=1)
        prediction_score = output_np[:, 0]
        print("\nPredicted privacy class:")
        if preds[0] == 0:
            print('Private')
        else:
            print('Public')
        print("Prediction score: ", prediction_score[0])
        print("Processing time: {:.4f}".format(time.time() - start_batch_time))
        return


if __name__ == '__main__':
    params.bbox_dir = params.root_dir + ''
    data_dir = params.root_dir + '/data_preprocess_'

    print("\nGraph-Privacy-Advisor Demo test\nfor image: ", params.image_name)
    evaluate(params.root_dir + 'graph_privacy_advisor/checkpoints/' + params.model_name, params.image_name)
