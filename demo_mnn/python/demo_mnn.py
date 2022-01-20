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
        pass

    @abstractmethod
    def infer_image(self, img_input):
        pass

    def detect(self, img):
        raw_shape = img.shape
        img_input, ResizeM = self.preprocess(img)
        output = self.infer_image(img_input)
        # code above this line is verified
        return output


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
        output = np.reshape(tmp_output.getData(), self.output_shape)
        return output


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



nanodet_mnn = NanoDetPlusMNN(model_path="nanodet-origin.mnn", num_classes=80)
nanodet_onnx = NanoDetPlusONNX(model_path="nanodet-origin.onnx", num_classes=80)
nanodet_torch = NanoDetPlusTorch(
    model_path="nanodet-plus-m_320_checkpoint.ckpt",
    cfg_path="config/nanodet-plus-m_320.yml",
)

cvimg = cv2.imread("demo_mnn/imgs/000258.jpg")

output_mnn = nanodet_mnn.detect(cvimg)
output_onnx = nanodet_onnx.detect(cvimg)
# output_torch = nanodet_torch.detect(cvimg)

assert np.max(output_mnn - output_onnx) < 1e-4
