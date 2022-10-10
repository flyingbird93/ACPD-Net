# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os, json
import scipy.io as scio
import numpy as np
import mmcv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images to coco format without annotations')
    parser.add_argument('img_path', default='/home/flyingbird/Work/Image_cropping/CACNet-Pytorch-main/dataset/FCDB/data/', help='The root path of images')
    parser.add_argument(
        'classes', type=str, default='class.txt', help='The text file name of storage class list')
    parser.add_argument(
        'out',
        type=str,
        default='coco_out.json',
        help='The output annotation json file name, The save dir is in the '
        'same directory as img_path')
    parser.add_argument(
        '-e',
        '--exclude-extensions',
        type=str,
        nargs='+',
        help='The suffix of images to be excluded, such as "png" and "bmp"')
    args = parser.parse_args()
    return args


def collect_image_infos(path, exclude_extensions=None):
    img_infos = []

    images_generator = mmcv.scandir(path, recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos


def FCDBDataset(split, FCDB_dir, keep_aspect_ratio=False):
    image_dir = os.path.join(FCDB_dir, 'data')
    assert os.path.exists(image_dir), image_dir
    # image_list = list(annos.keys())

    # def parse_annotations(self, split):
    if split == 'train':
        split_file = os.path.join(FCDB_dir, 'cropping_training_set.json')
    else:
        split_file = os.path.join(FCDB_dir, 'cropping_testing_set.json')
    assert os.path.exists(split_file), split_file
    origin_data = json.loads(open(split_file, 'r').read())
    annos = dict()
    for item in origin_data:
        url = item['url']
        image_name = os.path.split(url)[-1]
        if os.path.exists(os.path.join(image_dir, image_name)):
            x, y, w, h = item['crop']
            crop = [x, y, x + w, y + h]
            annos[image_name] = crop
    print('{} set, {} images'.format(split, len(annos)))
    return annos


def FLMSDataset(split, FLMS_dir, keep_aspect_ratio):
    data_dir = FLMS_dir
    assert os.path.exists(data_dir), data_dir
    image_dir = os.path.join(data_dir, 'image')
    assert os.path.exists(image_dir), image_dir

    image_crops_file = os.path.join(data_dir, '500_image_dataset.mat')
    assert os.path.exists(image_crops_file), image_crops_file
    image_crops = dict()
    anno = scio.loadmat(image_crops_file)
    for i in range(anno['img_gt'].shape[0]):
        image_name = anno['img_gt'][i, 0][0][0]
        gt_crops = anno['img_gt'][i, 0][1]
        gt_crops = gt_crops[:, [1, 0, 3, 2]]
        keep_index = np.where((gt_crops < 0).sum(1) == 0)
        gt_crops = gt_crops[keep_index].tolist()
        image_crops[image_name] = gt_crops
    print('{} images'.format(len(image_crops)))
    return image_crops


def cvt_to_coco_json(img_infos, classes):
    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        image_id += 1
    return coco


def main():
    args = parse_args()
    # assert args.out.endswith(
    #     'json'), 'The output file name must be json suffix'

    # 1 load image list info
    img_infos = collect_image_infos(args.img_path, args.exclude_extensions)

    # 2 convert to coco format data
    classes = mmcv.list_from_file(args.classes)
    print(classes)
    coco_info = cvt_to_coco_json(img_infos, classes)

    # 3 dump
    save_dir = os.path.join(args.img_path, '..', 'annotations')
    mmcv.mkdir_or_exist(save_dir)
    save_path = os.path.join(save_dir, args.out)
    mmcv.dump(coco_info, save_path)
    print(f'save json file: {save_path}')


if __name__ == '__main__':
    main()
