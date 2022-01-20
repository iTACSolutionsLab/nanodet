from hmac import digest_size
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
import MNN
import torch
import onnxruntime as ort
import math
from nanodet.data.transform.pipeline import Pipeline
from scipy.special import softmax

from nanodet.data.transform.warp import warp_boxes

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

# Copy from nanodet/data/transform/warp.py
def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """
    Get resize matrix for resizing raw img to input size
    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs


def overlay_bbox_cv(img, all_box, class_names):
    """Draw result boxes
    Copy from nanodet/util/visualization.py
    """
    # all_box array of [label, x0, y0, x1, y1, score]
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, score = box[0], box[5]
        x0, y0, x1, y1 = [int(i) for i in box[1:5]]
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img


class NanoDetPlusABC(metaclass=ABCMeta):
    def __init__(
        self,
        input_shape=[320, 320],
        img_mean=[103.53, 116.28, 123.675],
        img_std=[57.375, 57.12, 58.395],
        num_classes=1,
        reg_max=7,
        strides=[8, 16, 32, 64],
    ) -> None:
        pass
        self.input_shape = input_shape
        self.input_size = (self.input_shape[1], self.input_shape[0])
        self.img_mean = np.array(img_mean, dtype=np.float32).reshape(1, 1, 3) / 255
        self.img_std = np.array(img_std, dtype=np.float32).reshape(1, 1, 3) / 255
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.featmap_sizes = [
            (
                math.ceil(self.input_size[0] / stride),
                math.ceil(self.input_size[1]) / stride,
            )
            for stride in self.strides
        ]
        self.project = np.linspace(0, self.reg_max, self.reg_max + 1)
        self.nms_cfg = {"type": "nms", "iou_threshold": 0.6}
        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic_light",
            "fire_hydrant",
            "stop_sign",
            "parking_meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports_ball",
            "kite",
            "baseball_bat",
            "baseball_glove",
            "skateboard",
            "surfboard",
            "tennis_racket",
            "bottle",
            "wine_glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot_dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted_plant",
            "bed",
            "dining_table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell_phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy_bear",
            "hair_drier",
            "toothbrush",
        ]

    def preprocess(self, img: np.ndarray):
        # this is good
        # resize image
        ResizeM = get_resize_matrix(
            (img.shape[1], img.shape[0]), self.input_size, False
        )
        img_resize = cv2.warpPerspective(img, ResizeM, dsize=self.input_shape)
        # normalize image
        img_input = img_resize.astype(np.float32) / 255
        img_input = (img_input - self.img_mean) / self.img_std
        # expand dims
        img_input = np.transpose(img_input, [2, 0, 1])
        img_input = np.expand_dims(img_input, axis=0)
        return img_input, ResizeM

    def postprocess(self, preds, ResizeM):
        cls_scores, bbox_preds = np.split(preds, [self.num_classes], axis=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds)
        return result_list
        assert len(result_list) == 1, "batch size bigger than one is not supported"
        det_bboxes, det_labels = result_list[0]
        det_bboxes[:, :4] = warp_boxes(
            det_bboxes[:, :4],
            np.linalg.inv(ResizeM),
            self.input_shape[1],
            self.input_shape[0],
        )
        return det_bboxes[:, :4], det_bboxes[:, 4], det_labels

    def draw_box(self, raw_img, bbox, label, score):
        img = raw_img.copy()
        all_box = [
            [
                x,
            ]
            + y
            + [
                z,
            ]
            for x, y, z in zip(label, bbox.tolist(), score)
        ]
        img_draw = overlay_bbox_cv(img, all_box, self.class_names)
        return img_draw

    @abstractmethod
    def infer_image(self, img_input):
        pass

    def detect(self, img):
        raw_shape = img.shape
        img_input, ResizeM = self.preprocess(img)
        preds = self.infer_image(img_input)
        return self.postprocess(preds, ResizeM)
        bboxes, scores, labels = self.postprocess(preds, ResizeM)
        img_draw = self.draw_box(img, bboxes, labels, scores)
        cv2.imwrite("mnn_output.jpg", img_draw)

    def get_bboxes(self, cls_preds, reg_preds):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        b = cls_preds.shape[0]

        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(b, self.featmap_sizes[i], stride)
            for i, stride in enumerate(self.strides)
        ]
        center_priors = np.concatenate(mlvl_center_priors, axis=1)
        # code bofore this line is verified
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = self.distance2bbox(
            center_priors[..., :2], dis_preds, max_shape=self.input_size
        )
        # scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = cls_preds[i], bboxes[i]
            padding = np.zeros((score.shape[0], 1))
            score = np.concatenate([score, padding], axis=1)
            results = self.multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                max_num=100,
            )
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(self, batch_size, featmap_size, stride):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (np.arange(w)) * stride
        y_range = (np.arange(h)) * stride
        x, y = np.meshgrid(x_range, y_range)
        y = y.flatten()
        x = x.flatten()
        strides = np.full((x.shape[0],), stride)
        proiors = np.stack([x, y, strides, strides], axis=-1)
        ret = np.tile(np.expand_dims(proiors, axis=0), (batch_size, 1, 1))
        return ret

    def distribution_project(self, x):
        shape = x.shape
        x = softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), axis=-1)
        x = np.matmul(x, self.project).reshape(*shape[:-1], 4)
        return x

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def multiclass_nms(
        self,
        multi_bboxes,
        multi_scores,
        score_thr,
        max_num=-1,
    ):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the last column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_thr (float): NMS IoU threshold
            max_num (int): if there are more than max_num bboxes after NMS,
                only top max_num will be kept.
            score_factors (Tensor): The factors multiplied to scores before
                applying NMS

        Returns:
            tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
                are 0-based.
        """
        num_classes = multi_scores.shape[1] - 1
        # exclude background category
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.reshape(multi_scores.shape[0], -1, 4)
        else:
            bboxes = np.broadcast_to(
                multi_bboxes[:, None], (multi_scores.shape[0], num_classes, 4)
            )
        scores = multi_scores[:, :-1]

        # filter out boxes with low scores
        valid_mask = scores > score_thr

        # We use masked_select for ONNX exporting purpose,
        # which is equivalent to bboxes = bboxes[valid_mask]
        # we have to use this ugly code
        bboxes = bboxes[valid_mask].reshape(-1, 4)
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[1]

        if bboxes.size == 0:
            bboxes = np.zeros((0, 5))
            labels = np.zeros(0)
            return bboxes, labels

        dets, keep = self.batched_nms(bboxes, scores, labels, self.nms_cfg)
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]
        return dets, labels[keep]

    def batched_nms(self, boxes, scores, idxs, nms_cfg, class_agnostic=False):
        """Performs non-maximum suppression in a batched fashion.
        Modified from https://github.com/pytorch/vision/blob
        /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
        In order to perform NMS independently per class, we add an offset to all
        the boxes. The offset is dependent only on the class idx, and is large
        enough so that boxes from different classes do not overlap.
        Arguments:
            boxes (torch.Tensor): boxes in shape (N, 4).
            scores (torch.Tensor): scores in shape (N, ).
            idxs (torch.Tensor): each index value correspond to a bbox cluster,
                and NMS will not be applied between elements of different idxs,
                shape (N, ).
            nms_cfg (dict): specify nms type and other parameters like iou_thr.
                Possible keys includes the following.
                - iou_thr (float): IoU threshold used for NMS.
                - split_thr (float): threshold number of boxes. In some cases the
                    number of boxes is large (e.g., 200k). To avoid OOM during
                    training, the users could set `split_thr` to a small value.
                    If the number of boxes is greater than the threshold, it will
                    perform NMS on each group of boxes separately and sequentially.
                    Defaults to 10000.
            class_agnostic (bool): if true, nms is class agnostic,
                i.e. IoU thresholding happens over all boxes,
                regardless of the predicted class.
        Returns:
            tuple: kept dets and indice.
        """
        nms_cfg_ = nms_cfg.copy()
        class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
        if class_agnostic:
            boxes_for_nms = boxes
        else:
            max_coordinate = boxes.max()
            offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
            boxes_for_nms = boxes + offsets[:, None]
        nms_cfg_.pop("type", "nms")
        split_thr = nms_cfg_.pop("split_thr", 10000)
        # if len(boxes_for_nms) < split_thr:
        # keep = nms(boxes_for_nms, scores, **nms_cfg_)
        keep = self.hard_nms(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = scores[keep]
        # else:
        #     total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        #     for id in torch.unique(idxs):
        #         mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        #         keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
        #         total_mask[mask[keep]] = True

        #     keep = total_mask.nonzero(as_tuple=False).view(-1)
        #     keep = keep[scores[keep].argsort(descending=True)]
        #     boxes = boxes[keep]
        #     scores = scores[keep]
        return np.concatenate([boxes, scores[:, None]], -1), keep

    def hard_nms(self, boxes, scores, iou_threshold, top_k=-1, candidate_size=200):
        """

        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
            picked: a list of indexes of the kept boxes
        """
        picked = []
        # _, indexes = scores.sort(descending=True)
        indexes = np.argsort(scores)
        # indexes = indexes[:candidate_size]
        # indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            # current = indexes[0]
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            # indexes = indexes[1:]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return picked

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        """Return intersection-over-union (Jaccard index) of boxes.

        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def area_of(self, left_top, right_bottom):
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def warp_boxes(self, boxes, M, width, height):
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            )
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes


class NanoDetPlusMNN(NanoDetPlusABC):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Using MNN as inference backend")
        print(f"Using weight: {model_path}")

        dim = 0
        for stride in self.strides:
            dim += self.input_size[0] / stride * self.input_size[1] / stride
        self.output_shape = (1, int(dim), 4 * (self.reg_max + 1) + self.num_classes)
        # load model
        self.model_path = model_path
        self.interpreter = MNN.Interpreter(self.model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

    def infer_image(self, img_input):
        tmp_input = MNN.Tensor(
            (1, 3, self.input_size[1], self.input_size[0]),
            MNN.Halide_Type_Float,
            img_input,
            MNN.Tensor_DimensionType_Caffe,
        )
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)
        output_tensor = self.interpreter.getSessionOutput(self.session, "output")
        # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
        # reference: https://www.yuque.com/mnn/cn/vg3to5#KkxaS
        tmp_output = MNN.Tensor(
            self.output_shape,
            MNN.Halide_Type_Float,
            np.ones(self.output_shape).astype(np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        output_tensor.copyToHostTensor(tmp_output)
        preds = np.reshape(tmp_output.getData(), self.output_shape)
        return preds


class NanoDetPlusONNX(NanoDetPlusABC):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Using ONNX as inference backend")
        print(f"Using weight: {model_path}")

        # load model
        self.model_path = model_path
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def infer_image(self, img_input):
        output = self.ort_session.run(None, {self.input_name: img_input})[0]
        return output


class NanoDetPlusTorch(NanoDetPlusABC):
    def __init__(self, model_path, cfg_path, *args, **kwargs):
        from nanodet.model.arch import build_model
        from nanodet.util import Logger, cfg, load_config, load_model_weight

        super().__init__(*args, **kwargs)
        print("Using PyTorch as inference backend")
        print(f"Using weight: {model_path}")
        # load model
        self.model_path = model_path
        self.cfg_path = cfg_path
        load_config(cfg, cfg_path)
        self.logger = Logger(-1, cfg.save_dir, False)
        self.model = build_model(cfg.model)
        self.cfg = cfg
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(self.model, checkpoint, self.logger)
        self.device = "cuda:0"
        self.model = self.model.to(self.device).eval()

    def infer_image(self, img_input):
        self.model.train(False)
        with torch.no_grad():
            inference_results = self.model(torch.from_numpy(img_input))
        return inference_results.cpu().detach().numpy()

    def detect(self, img):
        from nanodet.data.collate import naive_collate
        from nanodet.data.batch_process import stack_batch_img

        raw_shape = img.shape
        pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)
        self.model.train(False)
        meta = {
            "img_info": {"height": raw_shape[0], "width": raw_shape[1]},
            "raw_img": img,
            "img": img,
        }
        meta = pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            return self.model.inference(meta)


# nanodet = NanoDetPlusMNN(model_path="nanodet.mnn")
# nanodet = NanoDetPlusONNX(model_path="nanodet.onnx")
# nanodet = NanoDetPlusTorch(model_path="model_best.ckpt", cfg_path="config/nanodet-plus-m_320_kondenai.yml")

# nanodet = NanoDetPlusMNN(model_path="nanodet-origin.mnn", num_classes=80)
# nanodet = NanoDetPlusONNX(model_path="nanodet-origin.onnx", num_classes=80)
# nanodet = NanoDetPlusTorch(model_path="nanodet-plus-m_320_checkpoint.ckpt", cfg_path="config/nanodet-plus-m_320.yml")


nanodet_mnn = NanoDetPlusMNN(model_path="nanodet-origin.mnn", num_classes=80)
# nanodet_onnx = NanoDetPlusONNX(model_path="nanodet-origin.onnx", num_classes=80)
nanodet_torch = NanoDetPlusTorch(
    model_path="nanodet-plus-m_320_checkpoint.ckpt",
    cfg_path="config/nanodet-plus-m_320.yml",
)

cvimg = cv2.imread("demo_mnn/imgs/000258.jpg")

output_mnn = nanodet_mnn.detect(cvimg)
# output_onnx = nanodet_onnx.detect(cvimg)
output_torch = nanodet_torch.detect(cvimg)
breakpoint()
