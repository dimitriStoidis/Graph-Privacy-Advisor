import torchvision.transforms as transforms
from data_preprocess_.dataset import IPDataset_FromFolder


def get_data_set(objects_dir, partition='train'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet values
    # scale_size = 256  # args.scale_size
    # crop_size = 224  # args.crop_size

    data_full_transform = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor(), normalize])  # what about horizontal flip

    data_set = IPDataset_FromFolder(objects_dir, data_full_transform, partition=partition)
    # returns target (also called labels), full_im (full sized original image without reshape), bboxes_14 (bbox
    # locations for max N objects (N can be 12 or 14). locations are resized to fit the 448x448 dim) , categories (of
    # objects found in bboxes), image_name # test_set = IPDataset(data_dir, objects_dir, test_full_transform)

    return data_set
