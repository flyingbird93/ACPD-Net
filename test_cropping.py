import torch
import os, json
import cv2, mmcv
import numpy as np
import scipy.io as scio
from PIL import Image

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class FCDBDataset(CustomDataset):
    CLASSES = ('background', 'crop_pred')

    # load anno
    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        # image_list = mmcv.list_from_file(self.ann_file)
        split_file = os.path.join(self.data_root, ann_file)
        origin_data = json.loads(open(split_file, 'r').read())
        annos = dict()
        for item in origin_data:
            url = item['url']
            image_name = os.path.split(url)[-1]
            if os.path.exists(os.path.join(self.img_prefix, image_name)):
                x,y,w,h = item['crop']
                crop = [x,y,x+w,y+h]
                annos[image_name] = crop

        data_infos = []
        # convert annotations to middle format
        for item in origin_data:
            img_id = os.path.split(item['url'])[-1]
            if os.path.exists(os.path.join(self.img_prefix, img_id)):
                filename = f'{self.img_prefix}/{img_id}'
                image = mmcv.imread(filename)
                height, width = image.shape[:2]

                data_info = dict(filename=f'{img_id}', width=width, height=height)

                x, y, w, h = item['crop']
                crop = [x, y, x + w, y + h]
                annos[image_name] = crop
                labels = np.ones((1,), dtype=np.compat.long)

                data_anno = dict(
                    bboxes=np.array(crop, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(labels, dtype=np.compat.long))

                data_info.update(ann=data_anno)
                data_infos.append(data_info)
        return data_infos


@DATASETS.register_module()
class FLMSDataset(CustomDataset):
    CLASSES = ('background', 'crop_pred')

    # load anno
    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        split_file = os.path.join(self.data_root, ann_file)
        anno = scio.loadmat(split_file)

        data_infos = []
        # convert annotations to middle format
        # for item in origin_data:
        for i in range(anno['img_gt'].shape[0]):
            image_name = anno['img_gt'][i, 0][0][0]
            # img_id = os.path.split(item['url'])[-1]
            if os.path.exists(os.path.join(self.img_prefix, image_name)):
                filename = f'{self.img_prefix}/{image_name}'
                image = mmcv.imread(filename)
                height, width = image.shape[:2]

                data_info = dict(filename=f'{image_name}', width=width, height=height)

                gt_crops = anno['img_gt'][i, 0][1]
                gt_crops = gt_crops[:, [1, 0, 3, 2]]
                keep_index = np.where((gt_crops < 0).sum(1) == 0)
                gt_crops = gt_crops[keep_index].tolist()
                # annos[image_name] = gt_crops
                labels = np.ones((len(gt_crops),), dtype=np.compat.long)

                data_anno = dict(
                    bboxes=np.array(gt_crops, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(labels, dtype=np.compat.long))

                data_info.update(ann=data_anno)
                data_infos.append(data_info)
        print('{} images'.format(len(data_infos)))
        return data_infos


@DATASETS.register_module()
class CUHKDataset(CustomDataset):
    CLASSES = ('background', 'crop_pred')

    # load anno
    def load_annotations(self, ann_file):
        # cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        # image_list = mmcv.list_from_file(self.ann_file)
        split_file = os.path.join(self.data_root, ann_file)
        origin_data = json.loads(open(split_file, 'r').read())

        # read anno
        all_file_list = []
        img_list = []
        anno_list = []
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                all_file_list.append(line.strip())

        for elem in range(0, len(all_file_list), 4):
            img_list.append(all_file_list[elem])
            anno_list.append(all_file_list[elem+1:elem+4])

        annos = dict()
        for item in range(len(img_list)):
            # url = item['url']
            image_name = img_list[item]
            if os.path.exists(os.path.join(self.img_prefix, image_name)):
                crop1 = [int(i) for i in anno_list[item][0].split(' ')]
                crop2 = [int(i) for i in anno_list[item][1].split(' ')]
                crop3 = [int(i) for i in anno_list[item][2].split(' ')]
                annos[image_name] = (crop1, crop2, crop3)

        data_infos = []
        # convert annotations to middle format
        for item in range(len(img_list)):
            img_id = img_list[item]
            if os.path.exists(os.path.join(self.img_prefix, img_id)):
                filename = f'{self.img_prefix}/{img_id}'
                image = mmcv.imread(filename)
                height, width = image.shape[:2]

                data_info = dict(filename=f'{img_id}', width=width, height=height)

                # num crop
                crop = annos[img_id][0]
                crop = [crop[0], crop[2], crop[1], crop[3]]
                labels = np.ones((1,), dtype=np.compat.long)

                data_anno = dict(
                    bboxes=np.array(crop, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(labels, dtype=np.compat.long))

                data_info.update(ann=data_anno)
                data_infos.append(data_info)
        return data_infos


# evaluate FCDB
def compute_iou_and_disp(gt_crop, pre_crop, img_w, img_h):
    '''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = torch.Tensor(gt_crop)# .unsqueeze(0)
    gt_crop = gt_crop.unsqueeze(0)

    # pre_crop = torch.Tensor(pre_crop[:, :4])  # .unsqueeze(0)
    pre_crop = torch.Tensor(pre_crop[:, :4])  # .unsqueeze(0)
    # pre_crop = pre_crop.unsqueeze(0)

    gt_crop = gt_crop[gt_crop[:,0] >= 0]
    zero_t  = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:,0], pre_crop[:,0])
    over_y1 = torch.maximum(gt_crop[:,1], pre_crop[:,1])
    over_x2 = torch.minimum(gt_crop[:,2], pre_crop[:,2])
    over_y2 = torch.minimum(gt_crop[:,3], pre_crop[:,3])
    over_w  = torch.maximum(zero_t, over_x2 - over_x1)
    over_h  = torch.maximum(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
    union   = area1 + area2 - inter
    iou     = inter / union
    disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / img_w + \
              (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / img_h
    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item(), index


# evaluate FLMS
def compute_iou_and_disp_FLMS(gt_crop, pre_crop, img_w, img_h):
    '''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = torch.Tensor(gt_crop)

    all_best_iou = []
    all_best_disp = []
    all_best_crop_index = []
    for crop in pre_crop:
        # print(crop)
        pre_crop = torch.Tensor(crop[:4])
        pre_crop = pre_crop.unsqueeze(0)

        # print(gt_crop.shape)
        # print(pre_crop)
        gt_crop = gt_crop[gt_crop[:,0] >= 0]
        zero_t  = torch.zeros(gt_crop.shape[0])
        over_x1 = torch.maximum(gt_crop[:,0], pre_crop[:,0])
        over_y1 = torch.maximum(gt_crop[:,1], pre_crop[:,1])
        over_x2 = torch.minimum(gt_crop[:,2], pre_crop[:,2])
        over_y2 = torch.minimum(gt_crop[:,3], pre_crop[:,3])
        over_w  = torch.maximum(zero_t, over_x2 - over_x1)
        over_h  = torch.maximum(zero_t, over_y2 - over_y1)
        inter   = over_w * over_h
        area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
        area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
        union   = area1 + area2 - inter
        iou     = inter / union
        disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / img_w + \
                  (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / img_h
        iou_idx = torch.argmax(iou, dim=-1)
        dis_idx = torch.argmin(disp, dim=-1)
        index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
        all_best_crop_index.append(index)
        all_best_iou.append(iou[index])
        all_best_disp.append(disp[index])

    # select best one
    iou_idx = torch.argmax(torch.Tensor(all_best_iou), dim=-1)
    dis_idx = torch.argmin(torch.Tensor(all_best_disp), dim=-1)
    best_index = dis_idx if (all_best_iou[iou_idx] == all_best_disp[dis_idx]) else iou_idx
    return all_best_iou[best_index].item(), all_best_disp[best_index].item(), best_index, all_best_crop_index[best_index]


def compute_iou_and_disp_FLMS_best_one(gt_crop, pre_crop, img_w, img_h):
    '''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = torch.Tensor(gt_crop)# .unsqueeze(0)

    pre_crop = torch.Tensor(pre_crop[0, :4])
    pre_crop = pre_crop.unsqueeze(0)
    # print(pre_crop.shape)

    # print(gt_crop.shape)
    # print(pre_crop)
    gt_crop = gt_crop[gt_crop[:,0] >= 0]
    zero_t  = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:,0], pre_crop[:,0])
    over_y1 = torch.maximum(gt_crop[:,1], pre_crop[:,1])
    over_x2 = torch.minimum(gt_crop[:,2], pre_crop[:,2])
    over_y2 = torch.minimum(gt_crop[:,3], pre_crop[:,3])
    over_w  = torch.maximum(zero_t, over_x2 - over_x1)
    over_h  = torch.maximum(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
    union   = area1 + area2 - inter
    iou     = inter / union
    disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / img_w + \
              (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / img_h
    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    # all_best_iou.append(iou[index])
    # all_best_disp.append(disp[index])

    # select best one
    # iou_idx = torch.argmax(torch.Tensor(all_best_iou), dim=-1)
    # dis_idx = torch.argmin(torch.Tensor(all_best_disp), dim=-1)
    # best_index = dis_idx if (all_best_iou[iou_idx] == all_best_disp[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item(), index


# evaluate CUHK
def compute_iou_and_disp_CUHK(gt_crop, pre_crop, im_w, im_h):
    '''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = torch.Tensor(gt_crop)# .unsqueeze(0)
    gt_crop = gt_crop.unsqueeze(0)
    # print(pre_crop.shape)
    pre_crop = torch.Tensor(pre_crop[:, :4])  # .unsqueeze(0)

    # print(pre_crop.shape)
    # print(gt_crop.shape)
    # print(pre_crop)
    gt_crop = gt_crop[gt_crop[:,0] >= 0]
    zero_t  = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:,0], pre_crop[:,0])
    over_y1 = torch.maximum(gt_crop[:,2], pre_crop[:,1])
    over_x2 = torch.minimum(gt_crop[:,1], pre_crop[:,2])
    over_y2 = torch.minimum(gt_crop[:,3], pre_crop[:,3])
    over_w  = torch.maximum(zero_t, over_x2 - over_x1)
    over_h  = torch.maximum(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,1] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,2])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:, 1])
    union   = area1 + area2 - inter
    iou     = inter / union
    disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 1] - pre_crop[:, 2])) / im_w + \
              (torch.abs(gt_crop[:, 2] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / im_h
    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item(), index


def evaluate_on_FCDB(pred_crop, gt_crop, img_w_list, img_h_list, file_path, img_root, save_results=True):
    #for index in range(2):
    # model.eval()
    # device = next(model.parameters()).device
    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0
    all_iou_list = []
    all_file_path = []
    crop_save_dir = 'results/crop_result/'
    crop_result_dir = 'results/cropped_result/'

    for elem in range(len(gt_crop)):
        # print(gt_crop[elem].shape)
        iou, disp, img_index = compute_iou_and_disp(gt_crop[elem], pred_crop[elem][1], img_w_list[elem], img_h_list[elem])
        all_iou_list.append(iou)
        all_file_path.append(file_path[elem])
        if iou >= alpha:
            alpha_cnt += 1
        accum_iou += iou
        accum_disp += disp
        cnt += 1

        if save_results:
            # print(pred_crop[elem][index])
            # best_crop = pred_crop[elem].numpy().tolist()
            best_crop = pred_crop[elem][1][img_index, :4]
            best_crop = [int(x) for x in best_crop] # x1,y1,x2,y2
            print(best_crop)

            gt_crop_elem = gt_crop[elem]
            gt_crop_range = [int(x) for x in gt_crop_elem]
            # print(gt_crop_range)
            # test_results[image_name] = best_crop

            # save the best crop
            source_img = cv2.imread(img_root + file_path[elem])
            # print(source_img.shape)
            # croped_img  = source_img_1[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            # label1 = 'Pred'
            # cropping_img = cv2.rectangle(source_img_1, (best_crop[0], best_crop[1]), (best_crop[2], best_crop[3]), (255,0,255), 5)
            # font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            # imgzi1 = cv2.putText(cropping_img, '{}'.format(label1), (best_crop[2]-40, best_crop[3]-10), font, 0.5, (255, 0, 255), 1)
            # label2 = 'GT'
            # cropping_img = cv2.rectangle(cropping_img, (gt_crop_range[0], gt_crop_range[1]), (gt_crop_range[2], gt_crop_range[3]), (0, 255, 255), 5)
            # font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            # imgzi2 = cv2.putText(cropping_img, '{}'.format(label2), (gt_crop_range[2]-40, gt_crop_range[3]-10), font, 0.5, (0, 255, 255), 1)
            # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
            cropped_img = source_img[best_crop[1]:best_crop[3], best_crop[0]:best_crop[2], :]
            # cv2.imwrite(os.path.join(crop_save_dir, file_path[elem]), cropping_img)
            # print(os.path.join(crop_result_dir, file_path[elem]))
            cv2.imwrite(os.path.join(crop_result_dir, file_path[elem]), cropped_img)
            # break
    # if save_results:
    #     with open(save_file, 'w') as f:
    #         json.dump(test_results, f)
    avg_iou  = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp, all_iou_list, all_file_path


def evaluate_on_FCDB_best_one(pred_crop, gt_crop, img_w_list, img_h_list, file_path, img_root, save_results=False):
    # model.eval()
    # device = next(model.parameters()).device
    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0
    crop_save_dir = 'results/crop_result/'

    for elem in range(len(gt_crop)):
        # print(gt_crop[elem].shape)
        iou, disp, img_index = compute_iou_and_disp(gt_crop[elem], pred_crop[elem][1][0], img_w_list[elem], img_h_list[elem], FLMS=False)
        if iou >= alpha:
            alpha_cnt += 1
        accum_iou += iou
        accum_disp += disp
        cnt += 1

        if save_results:
            # print(pred_crop[elem][index])
            # best_crop = pred_crop[elem].numpy().tolist()
            best_crop = pred_crop[elem][1][0, :4]
            best_crop = [int(x) for x in best_crop] # x1,y1,x2,y2

            gt_crop_elem = gt_crop[elem]
            gt_crop_range = [int(x) for x in gt_crop_elem]
            # print(gt_crop_range)
            # test_results[image_name] = best_crop

            # save the best crop
            source_img = cv2.imread(img_root + file_path[elem])
            # croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            label1 = 'Pred'
            cropping_img = cv2.rectangle(source_img, (best_crop[0], best_crop[1]), (best_crop[2], best_crop[3]), (255,0,255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi1 = cv2.putText(cropping_img, '{}'.format(label1), (best_crop[2]-40, best_crop[3]-10), font, 0.5, (255, 0, 255), 1)
            label2 = 'GT'
            cropping_img = cv2.rectangle(cropping_img, (gt_crop_range[0], gt_crop_range[1]), (gt_crop_range[2], gt_crop_range[3]), (0, 255, 255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi2 = cv2.putText(cropping_img, '{}'.format(label2), (gt_crop_range[2]-40, gt_crop_range[3]-10), font, 0.5, (0, 255, 255), 1)
            # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
            cv2.imwrite(os.path.join(crop_save_dir, file_path[elem]), cropping_img)
            # break
    # if save_results:
    #     with open(save_file, 'w') as f:
    #         json.dump(test_results, f)
    avg_iou  = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp


def evaluate_on_FLMS(pred_crop, gt_crop, FLMS_img_w_list, FLMS_img_h_list, file_path, img_root, save_results=True):
    # for index in range(2):
    # model.eval()
    # device = next(model.parameters()).device
    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0
    crop_save_dir = 'results/FLMS_result/'
    cropped_save_dir = 'results/FLMS_cropped_result/'

    for elem in range(len(gt_crop)):
        # print(pred_crop[elem][0].shape)
        # gt_crop[elem] 1, 10, 4
        # pred_crop[[elem] 2, rand, 5
        # print(len(pred_crop))   # 500, nms_num, (n, 5)
        # print(len(pred_crop[0]))
        # print(len(pred_crop[0][0]))
        # print(pred_crop.shape)
        iou, disp, crop_index, gt_index = compute_iou_and_disp_FLMS(gt_crop[elem], pred_crop[elem][1], FLMS_img_w_list[elem], FLMS_img_h_list[elem])
        if iou >= alpha:
            alpha_cnt += 1
        accum_iou += iou
        accum_disp += disp
        cnt += 1
        # iou, disp = compute_iou_and_disp(torch.Tensor(gt_crop[elem]), torch.Tensor(pred_crop[elem][0]), img_w_list[elem], img_h_list[elem], FLMS=True)
        # if iou >= alpha:
        #     alpha_cnt += 1
        # accum_iou += iou
        # accum_disp += disp
        # cnt += 1

        if save_results:
            best_crop = pred_crop[elem][1][crop_index, :4]
            best_crop = [int(x) for x in best_crop]  # x1,y1,x2,y2

            gt_crop_elem = gt_crop[elem][gt_index]
            gt_crop_range = [int(x) for x in gt_crop_elem]
            # print(gt_crop_range)
            # test_results[image_name] = best_crop

            # save the best crop
            source_img = cv2.imread(img_root + file_path[elem])
            # croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            label1 = 'Pred'
            cropping_img = cv2.rectangle(source_img, (best_crop[0], best_crop[1]), (best_crop[2], best_crop[3]),
                                         (255, 0, 255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi1 = cv2.putText(cropping_img, '{}'.format(label1), (best_crop[2] - 40, best_crop[3] - 10), font, 0.5,
                                 (255, 0, 255), 1)
            label2 = 'GT'
            cropping_img = cv2.rectangle(cropping_img, (gt_crop_range[0], gt_crop_range[1]),
                                         (gt_crop_range[2], gt_crop_range[3]), (0, 255, 255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi2 = cv2.putText(cropping_img, '{}'.format(label2), (gt_crop_range[2] - 40, gt_crop_range[3] - 10),
                                 font, 0.5, (0, 255, 255), 1)
            # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
            cv2.imwrite(os.path.join(crop_save_dir, file_path[elem]), cropping_img)

            # best_crop = pred_crop[0].numpy().tolist()
            # best_crop = [int(x) for x in best_crop] # x1,y1,x2,y2
            # # test_results[image_name] = best_crop
            #
            # # save the best crop
            source_img = cv2.imread(img_root + file_path[elem])
            croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            cv2.imwrite(os.path.join(cropped_save_dir, file_path[elem]), croped_img)
    # if save_results:
    #     with open(save_file, 'w') as f:
    #         json.dump(test_results, f)
    avg_iou  = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp


def evaluate_on_FLMS_best_one(pred_crop, gt_crop, FLMS_img_w_list, FLMS_img_h_list, file_path, img_root, save_results=False):
    # for index in range(2):
    # model.eval()
    # device = next(model.parameters()).device
    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0
    crop_save_dir = 'results/crop_result/'

    for elem in range(len(gt_crop)):
        # print(pred_crop[elem][0].shape)
        # gt_crop[elem] 1, 10, 4
        # pred_crop[[elem] 2, rand, 5
        # print(len(pred_crop))   # 500, nms_num, (n, 5)
        # print(len(pred_crop[0]))
        # print(len(pred_crop[0][0]))
        # print(pred_crop.shape)
        iou, disp, img_index = compute_iou_and_disp_FLMS_best_one(gt_crop[elem], pred_crop[elem][1], FLMS_img_w_list[elem], FLMS_img_h_list[elem])
        if iou >= alpha:
            alpha_cnt += 1
        accum_iou += iou
        accum_disp += disp
        cnt += 1
        # iou, disp = compute_iou_and_disp(torch.Tensor(gt_crop[elem]), torch.Tensor(pred_crop[elem][0]), img_w_list[elem], img_h_list[elem], FLMS=True)
        # if iou >= alpha:
        #     alpha_cnt += 1
        # accum_iou += iou
        # accum_disp += disp
        # cnt += 1

        if save_results:
            best_crop = pred_crop[elem][1][0, :4]
            best_crop = [int(x) for x in best_crop]  # x1,y1,x2,y2

            gt_crop_elem = gt_crop[elem]
            gt_crop_range = [int(x) for x in gt_crop_elem]
            # print(gt_crop_range)
            # test_results[image_name] = best_crop

            # save the best crop
            source_img = cv2.imread(img_root + file_path[elem])
            # croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            label1 = 'Pred'
            cropping_img = cv2.rectangle(source_img, (best_crop[0], best_crop[1]), (best_crop[2], best_crop[3]),
                                         (255, 0, 255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi1 = cv2.putText(cropping_img, '{}'.format(label1), (best_crop[2] - 40, best_crop[3] - 10), font, 0.5,
                                 (255, 0, 255), 1)
            label2 = 'GT'
            cropping_img = cv2.rectangle(cropping_img, (gt_crop_range[0], gt_crop_range[1]),
                                         (gt_crop_range[2], gt_crop_range[3]), (0, 255, 255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi2 = cv2.putText(cropping_img, '{}'.format(label2), (gt_crop_range[2] - 40, gt_crop_range[3] - 10),
                                 font, 0.5, (0, 255, 255), 1)
            # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
            cv2.imwrite(os.path.join(crop_save_dir, file_path[elem]), cropping_img)

            # best_crop = pred_crop[0].numpy().tolist()
            # best_crop = [int(x) for x in best_crop] # x1,y1,x2,y2
            # # test_results[image_name] = best_crop
            #
            # # save the best crop
            # source_img = cv2.imread(img_root + file_path[elem])
            # croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            # cv2.imwrite(os.path.join(crop_save_dir, file_path[elem]), croped_img)
    # if save_results:
    #     with open(save_file, 'w') as f:
    #         json.dump(test_results, f)
    avg_iou  = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp


def evaluate_on_CUHK(pred_crop, gt_crop, img_w_list, img_h_list, file_path, img_root, save_results=False):
    # model.eval()
    # device = next(model.parameters()).device
    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0
    crop_save_dir = 'results/CUHK_result/'

    for elem in range(len(gt_crop)):
        # print(pred_crop[elem][0].shape)
        # gt_crop[elem] 1, 10, 4
        # pred_crop[[elem] 2, rand, 5
        iou, disp, index = compute_iou_and_disp_CUHK(gt_crop[elem], pred_crop[elem][0], img_w_list[elem], img_h_list[elem])
        if iou >= alpha:
            alpha_cnt += 1
        accum_iou += iou
        accum_disp += disp
        cnt += 1

        if save_results:
            best_crop = pred_crop[elem][1][index, :4]
            best_crop = [int(x) for x in best_crop]  # x1,y1,x2,y2

            gt_crop_elem = gt_crop[elem]
            gt_crop_range = [int(x) for x in gt_crop_elem]
            # print(gt_crop_range)
            # test_results[image_name] = best_crop

            # save the best crop
            source_img = cv2.imread(img_root + file_path[elem])
            # croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            label1 = 'Pred'
            cropping_img = cv2.rectangle(source_img, (best_crop[0], best_crop[1]), (best_crop[2], best_crop[3]),
                                         (255, 0, 255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi1 = cv2.putText(cropping_img, '{}'.format(label1), (best_crop[2] - 40, best_crop[3] - 10), font, 0.5,
                                 (255, 0, 255), 1)
            label2 = 'GT'
            cropping_img = cv2.rectangle(cropping_img, (gt_crop_range[0], gt_crop_range[1]),
                                         (gt_crop_range[2], gt_crop_range[3]), (0, 255, 255), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi2 = cv2.putText(cropping_img, '{}'.format(label2), (gt_crop_range[2] - 40, gt_crop_range[3] - 10),
                                 font, 0.5, (0, 255, 255), 1)
            # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
            cv2.imwrite(os.path.join(crop_save_dir, file_path[elem]), cropping_img)

    avg_iou  = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp


# load test anno
def parse_annotations(data_dir, image_dir):
    img_h = []
    img_w = []
    split_file = os.path.join(data_dir, 'cropping_testing_set.json')
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

            image_file = os.path.join(image_dir, image_name)
            image = Image.open(image_file).convert('RGB')
            im_width, im_height = image.size
            img_h.append(im_height)
            img_w.append(im_width)
    print('test set, {} images'.format(len(annos)))
    return annos, img_w, img_h


# load test anno
def parse_CUHK_annotations(data_dir, image_dir):
    img_h = []
    img_w = []
    all_file_list = []
    img_list = []
    anno_list = []

    file_path = os.path.join(data_dir, 'Cropping_parameters.txt')
    with open(file_path, 'r') as f:
        for line in f.readlines():
            all_file_list.append(line.strip())

    for elem in range(0, len(all_file_list), 4):
        img_list.append(all_file_list[elem])
        anno_list.append(all_file_list[elem + 1:elem + 4])

    annos = dict()
    for item in range(len(img_list)):
        # print(img_list[item])
        image_name = img_list[item].split('\\')[-1]
        if os.path.exists(os.path.join(image_dir, image_name)):
            crop1 = [int(i) for i in anno_list[item][0].split(' ')]
            crop2 = [int(i) for i in anno_list[item][1].split(' ')]
            crop3 = [int(i) for i in anno_list[item][2].split(' ')]
            annos[image_name] = (crop1, crop2, crop3)

            image_file = os.path.join(image_dir, image_name)
            image = Image.open(image_file).convert('RGB')
            im_width, im_height = image.size
            img_h.append(im_height)
            img_w.append(im_width)
    print('test set, {} images'.format(len(annos)))
    return annos, img_w, img_h


def parse_FLMS_annotations(data_dir, image_dir):
    img_h = []
    img_w = []
    split_file = os.path.join(data_dir, '500_image_dataset.mat')
    assert os.path.exists(split_file), split_file
    origin_data = scio.loadmat(split_file)
    annos = dict()
    for i in range(origin_data['img_gt'].shape[0]):
        image_name = origin_data['img_gt'][i, 0][0][0]
        image_file = os.path.join(image_dir, image_name)
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        img_h.append(im_height)
        img_w.append(im_width)

        gt_crops = origin_data['img_gt'][i, 0][1]
        gt_crops = gt_crops[:, [1, 0, 3, 2]]
        keep_index = np.where((gt_crops < 0).sum(1) == 0)
        gt_crops = gt_crops[keep_index].tolist()
        annos[image_name] = gt_crops
    print('test FLMS set, {} images'.format(len(annos)))
    return annos, img_w, img_h