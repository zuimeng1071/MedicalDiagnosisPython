import logging
import os

import torch

from models.DASPP_ChannelAtte_UNet import DASPP_ChannelAtte_UNet
from models.Unet import Unet
from .ImageUtils import ImageUtils


class Segment:

    @staticmethod
    def load_model(model_name: str, image_size=(400, 400), isLoadWeight=False, weight_path=None):
        """
        加载模型函数
        :param model_name: 需要加载的模型 包含 Unet、DASPPChannelAtteUnet
        :param image_size: 输入到模型中的图片尺寸，部分模型有要求的最小尺寸
        :param isLoadWeight: 是否加载原有的权重
        :param weight_path: 模型权重文件路径
        :return: 加载好的模型
        """
        # 检查图片尺寸是否符合要求
        if image_size[0] % 16 != 0 and image_size[1] % 16 != 0:
            logging.log(logging.WARNING, "图片尺寸不等于16的倍数，这可能会导致模型报错")

        # 根据模型名称初始化模型实例
        if model_name == 'Unet':
            model = Unet()
        elif model_name == 'DASPPChannelAtteUnet':
            model = DASPP_ChannelAtte_UNet()
        else:
            logging.log(logging.WARNING, f"模型名称错误:{model_name}")
            return None

        # 如果模型未成功初始化则返回None
        if model is None:
            return None

        # 如果需要加载权重文件，则尝试从指定路径加载
        if isLoadWeight:
            if os.path.exists(weight_path):
                model.load_state_dict(torch.load(weight_path))
                logging.log(logging.INFO, f"加载权重文件:{weight_path}")
            else:
                logging.log(logging.WARNING, f"权重文件不存在:{weight_path}")
                return None

        return model

    @staticmethod
    def predict(model: torch.nn.Module, inputTensor: torch.Tensor):
        """
        :param model: 模型实例
        :param inputTensor: (C, H, W) 的 tensor
        :return: outArr: (H, W) 的 uint8 数组，值为 0 或 255
        """
        if model is None:
            return None, None

        model.eval()
        with torch.no_grad():
            # inputTensor 是 (C, H, W)，需要加 batch 维度
            inputTensor = inputTensor.unsqueeze(0)  # -> (1, C, H, W)
            outputTensor = model(inputTensor)  # -> (1, 1, H, W)
            outputTensor = outputTensor.squeeze(0)  # -> (1, H, W)

            _, outArr = ImageUtils.imagePostProcessing(outputTensor)  # outArr: (H, W)

        return outArr, outputTensor  # 返回 (H, W) 数组
