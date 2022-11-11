import json
import numpy
from yolo_models import *
from utils import load_classes, non_max_suppression
import cv2
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

"""
Darknet model originally from https://pjreddie.com/darknet/yolo/
"""

root_dir = ''
config_path = root_dir + '/data_preprocess_/config/yolov3.cfg'
weights_path = root_dir + '/data_preprocess_/config/yolov3.weights'
class_path = root_dir + '/data_preprocess_/config/coco.names'
# hyper-parameters in yolov3
img_size = 416
conf_thres = 0.8
nms_thres = 0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor


def generate_json(pred_cat, bboxes, filename, _set, database):
    bboxes_categories = {"categories": [], "bboxes": []}
    bboxes_categories["categories"].append(pred_cat)
    bboxes_categories["bboxes"].append(bboxes)
    bboxes_categories["categories"] = bboxes_categories["categories"][0]
    bboxes_categories["bboxes"] = bboxes_categories["bboxes"][0]
    json_file = json.dumps(bboxes_categories)
    if not os.path.exists('./' + _set + database + 'bboxes/'):
        os.mkdir('./' + _set + database + 'bboxes/')
    with open('./' + _set + database + 'bboxes/' + filename + '.json', 'w') as f:
        f.write(json_file)


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2), 0), max(int((imw-imh)/2), 0), max(int((imh-imw)/2), 0), max(int((imw-imh)/2),0)),
                        (128, 128, 128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)

    return detections[0]


if __name__ == '__main__':
    _set = ['train_', 'test_']
    database = 'PicAlert'
    for s in _set:
        print("Detections for Set: ", s)
        public_file = open(database + s + 'public_files_path.txt', 'r')
        pub_file = public_file.readlines()
        private_file = open(database + s + 'private_files_path.txt', 'r')
        priv_file = private_file.readlines()

        temp_file = open(database + s + 'files.txt', 'w')
        [temp_file.write(file1) for file1 in pub_file]
        [temp_file.write(file2) for file2 in priv_file]
        temp_file.close()
        temp_file = open(database + s + 'files.txt', 'r')
        images = temp_file.readlines()
        prev_time = time.time()

        print("Number of images Images:{}".format(len(images)))
        boundingBox = []

        count = 0
        detectedObjectsFile = open(s + database + 'bboxes.csv', 'w')
        writer = csv.writer(detectedObjectsFile)

        for i, file in enumerate(images):
            filename = file.split("/")[-1].split(".")[0]
            try:
                img = Image.open(file.strip()).convert('RGB')
                detections = detect_image(img)
                print("image ", filename)

                img = np.array(img)
                pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
                unpad_h = img_size - pad_y
                unpad_w = img_size - pad_x
                categs, obj_categ, bboxes = [], [], []

                if detections is not None:
                    print("Detections...")
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)

                    # browse detections and draw bounding boxes
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                        bb = [x1.item(), y1.item(), box_w.item(), box_h.item()]  # bb = [x1.item(), y1.item(), x2.item(), y2.item()]
                        bboxes.append(bb)
                        categs.append(int(cls_pred))
                        obj_categ.append(classes[int(cls_pred)])
                        print(classes[int(cls_pred)])
                    row = [filename, obj_categ]
                    writer.writerow(row)

                    generate_json(categs, bboxes, filename, s, database)
                else:
                    print("Background \n")
                    row = [filename, 'background']
                    writer.writerow(row)
                    bb = [[0, 0, img.shape[0], img.shape[1]]]
                    generate_json([80], bb, filename, s, database)
            except:
                pass

            print('[%] {:.3f}'.format((i * 100) / len(images)))
        detectedObjectsFile.close()

        inference_time = datetime.timedelta(seconds=time.time() - prev_time)
        print('Inference Time: %s' % inference_time)