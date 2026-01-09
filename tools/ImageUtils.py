import base64
import logging

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from sklearn.decomposition import PCA


class ImageUtils:
    """
    图像处理工具类
    """

    def __init__(self):
        pass

    @staticmethod
    def padAndResize(imgArr: np.ndarray, n=400):
        """
        将h*w*c的图片转为 max(h,w),max(h,w),c大小的图片, 并缩放为n,n,c
        :param imgArr: 图片矩阵
        :param n: 缩放后的图片大小
        :return: 缩放后的图片矩阵
        """
        max_size = max(imgArr.shape[:-1])
        imgArr = np.pad(imgArr, ((0, max_size - imgArr.shape[0]), (0, max_size - imgArr.shape[1]), (0, 0)),
                        'reflect')
        imgArr = np.array(Image.fromarray(imgArr).resize((n, n)))

        return imgArr

    @staticmethod
    def resizeBackToOriginal(
            square_img_arr: np.ndarray,
            originalSize: tuple[int, int],
            resample=Image.Resampling.BILINEAR
    ) -> np.ndarray:
        """
        将 n,n,c 的正方形图像缩放并裁剪回原始的 h,w,c 尺寸。

        :param square_img_arr: 输入的正方形图像数组，形状为 (n, n) 或 (n, n, c)
        :param originalSize: 原始图像的尺寸 (height, width)，即 (h, w)
        :param resample: PIL 重采样方法，对于掩码推荐使用 NEAREST，对于图像可用 BILINEAR
        :return: 恢复后的图像数组，形状为 (h, w) 或 (h, w, c)
        """
        h, w = originalSize
        max_dim = max(h, w)

        # 1: 将 n*n 的图像 resize 回 max_dim * max_dim 的正方形
        img_pil = Image.fromarray(square_img_arr)
        resized_square = img_pil.resize((max_dim, max_dim), resample=resample)  # 注意：PIL 使用 (宽, 高)，但这里正方形无所谓
        resizedArr = np.array(resized_square)

        # 2: 裁剪到原始的 (h, w) 尺寸（左上对齐裁剪，与 pad 的方式对应）
        # 由于 pad 是在右下填充，所以裁剪时取左上角 [0:h, 0:w]
        restored = resizedArr[:h, :w]

        return restored

    @staticmethod
    def applyPca(imageArr: np.ndarray, n_components=64):
        """
        对图像进行PCA降维
        :param imageArr: 图像矩阵
        :param n_components: 降维后的维数
        :return: 降维后的图像矩阵
        """
        # 将三个(R,G,B)通道简单拼接在一起
        im1 = np.hstack((imageArr[:, :, 0], imageArr[:, :, 1], imageArr[:, :, 2]))

        # 对每个通道进行归一化处理，使每列的均值为0，标准差为1
        means = np.mean(im1, axis=0)
        sds = np.std(im1, axis=0)

        # 防止除以0，给标准差加一个小的常数
        epsilon = 1e-8
        sds[sds == 0] = epsilon

        im2 = (im1 - means) / sds

        # 使用PCA进行降维与重构
        pca = PCA(n_components=n_components)

        # 计算原维度和降维后的维度
        C = pca.fit_transform(im2)  # 进行PCA变换

        # 重构数据
        im3 = pca.inverse_transform(C)

        # 反归一化
        im3 = np.clip(im3 * sds + means, 0, 255)
        im3 = im3.astype('uint8')

        # 重新分割成三个(R,G,B)通道
        im3_channels = np.hsplit(im3, 3)
        im4 = np.zeros_like(imageArr)
        for i in range(3):
            im4[:, :, i] = im3_channels[i]

        return im4.astype('uint8')

    @staticmethod
    def imageArrayToTensorWithEnhance(image_arr: np.ndarray,
                                      mean=None,
                                      std=None):
        """
        将输入的 RGB 图像数组（H, W, C）转换为经过预处理的 PyTorch 张量（C, H, W）

        包含：CLAHE增强、中值滤波、伽马校正、归一化、ToTensor

        :param image_arr: 输入图像，numpy array，形状 (H, W, 3)，RGB格式，uint8 [0,255]
        :param mean: 归一化均值，tuple 或 list，长度为3
        :param std: 归一化标准差，tuple 或 list，长度为3
        :return: 处理后的 PyTorch Tensor，形状 (3, H, W)，在CPU上
        """

        # 定义 Albumentations 预处理 pipeline
        if std is None:
            std = [0.5, 0.33, 0.33]
        if mean is None:
            mean = [0.45, 0.5, 0.5]
        transform = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),  # 输出为 (C, H, W) 的 torch.Tensor
        ])

        # 应用变换（注意：albumentations 接收 dict）
        transformed = transform(image=image_arr)
        tensor = transformed["image"]  # shape: (C, H, W)

        return tensor

    @staticmethod
    def imagePostProcessing(inputDatas: torch.Tensor,
                            threshold=0.35,
                            kernelSize=15,
                            iterations=2):
        """
        输入: (B, C, H, W) 或 (C, H, W)
        输出: processed_tensor (B,1,H,W) 和 processed_arr (H, W) 单通道
        """
        inputDatas = inputDatas.float()
        if inputDatas.min() < 0 or inputDatas.max() > 1:
            raise ValueError("输入应该在[0, 1]之间")

        processed_datas = []

        # 确保是4D: (B, C, H, W)
        if inputDatas.dim() == 3:
            inputDatas = inputDatas.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        if inputDatas.shape[1] != 1:
            raise ValueError("输入通道数应为1")

        for input_data in inputDatas:  # input_data: (1, H, W)
            # squeeze channel
            numpyData = input_data.squeeze(0).cpu().numpy()  # -> (H, W)

            # 二值化
            binaryData = (numpyData > threshold).astype(np.uint8) * 255
            kernel = np.ones((kernelSize, kernelSize), np.uint8)

            # 形态学操作
            closedData = cv2.morphologyEx(binaryData, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            opened_data = cv2.morphologyEx(closedData, cv2.MORPH_OPEN, kernel, iterations=iterations)

            processed_datas.append(opened_data)

        # 堆叠成 (B, H, W)
        processed_arr = np.array(processed_datas)  # shape: (B, H, W)

        # 如果 batch=1，只返回 (H, W)
        if processed_arr.shape[0] == 1:
            processed_arr = processed_arr[0]  # -> (H, W)

        # 构造返回的 tensor (保持 float [0,1])
        processed_tensor = torch.from_numpy(processed_arr).to(inputDatas.device).float() / 255.0
        if len(processed_tensor.shape) == 2:
            processed_tensor = processed_tensor.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)

        return processed_tensor, processed_arr

    @staticmethod
    def mergeMaskAndOriginal(
            originalImgArr: np.ndarray,
            maskImgArr: np.ndarray,
            alpha: float = 0.5,
            color: tuple[int, int, int] = (0, 255, 0)  # 默认绿色
    ) -> np.ndarray:
        """
        合并原始图像和掩码图像

        :param originalImgArr: 原始图像矩阵(HWC)，dtype=uint8，RGB格式
        :param maskImgArr: 掩码图像矩阵(HW)，dtype=uint8，单通道，[0, 255]
        :param alpha: 原图透明度权重，范围[0,1]，值越大原图越明显
        :param color: 掩码颜色，默认为绿色(0, 255, 0)
        :return: 合并后的图像矩阵(HWC)，dtype=uint8，RGB格式
        """
        if alpha < 0 or alpha > 1:
            logging.log(logging.WARNING, "alpha参数范围应该在[0,1]之间")
        elif originalImgArr.shape[:2] != maskImgArr.shape[:2]:
            logging.log(logging.WARNING, "原始图像和掩码图像的尺寸不一致")
        elif originalImgArr.shape[2] != 3:
            logging.log(logging.WARNING, "原始图像的通道数应该为3")

        # 确保mask是二值化的并且为单通道
        if maskImgArr.dtype != np.uint8 or len(maskImgArr.shape) > 2:
            logging.log(logging.WARNING, "掩码图像应该为单通道的uint8格式")
        mask_colored = np.zeros_like(originalImgArr)
        # 使用指定的颜色填充掩码区域
        mask_colored[maskImgArr > 0] = color

        # 根据alpha混合原始图像和掩码颜色
        mergedImage = cv2.addWeighted(originalImgArr, 1, mask_colored, alpha, 0)
        mergedImage = cv2.cvtColor(mergedImage, cv2.COLOR_BGR2RGB)

        return mergedImage

    @staticmethod
    def ImagePreProcessing(imageArr: np.ndarray,
                           imageSize: int = 400,
                           mean=None,
                           std=None) -> torch.Tensor:
        """
        对输入的图像进行预处理，包括填充为正方形、应用PCA和基础图像增强
        :param imageArr: 输入的图像矩阵，形状为(H, W, C)，dtype为uint8
        :param imageSize: 输出图像的大小，默认为400
        :param mean: 均值，用于归一化，默认为[0.5, 0.33, 0.33]
        :param std: 标准差，用于归一化，默认为[0.45, 0.5, 0.5]
        :return: 处理后的图像张量，形状为(C, H, W)，dtype为float32
        """
        # 1. 填充为正方形
        imageArr = ImageUtils.padAndResize(imageArr, imageSize)
        # 2. 应用PCA
        imageArr = ImageUtils.applyPca(imageArr)
        # 3. 应用基础图像增强
        imageTensor = ImageUtils.imageArrayToTensorWithEnhance(imageArr, mean, std)
        return imageTensor

    @staticmethod
    def tensorToArray(imageTensor: torch.Tensor) -> np.ndarray:
        """
        将张量转换为数组

        :param imageTensor: 输入的图像张量，形状为(C, H, W)，dtype为float32
        :return: 数组，形状为(H, W, C)，dtype为uint8
        """
        # 将张量移动到CPU并转换为NumPy数组
        array = imageTensor.cpu().detach().numpy()

        # 如果数据是单通道灰度图像，则直接扩展维度
        if array.shape[0] == 1:
            array = array.squeeze(axis=0)
        else:
            # 对于多通道图像，交换维度使通道位于最后 (C, H, W) -> (H, W, C)
            array = np.transpose(array, (1, 2, 0))

        # 归一化处理，确保数据在 [0, 255] 范围内，并转换为 uint8 类型
        array = (array - array.min()) / (array.max() - array.min()) * 255.0
        array = array.clip(0, 255).astype(np.uint8)

        return array


    @staticmethod
    def ImagePreProcessing2Arr(imageArr: np.ndarray,
                           imageSize: int = 400,
                           mean=None,
                           std=None) -> np.ndarray:
        """
        对输入的图像进行预处理，包括填充为正方形、应用PCA和基础图像增强
        :param imageArr: 输入的图像矩阵，形状为(H, W, C)，dtype为uint8
        :param imageSize: 输出图像的大小，默认为400
        :param mean: 均值，用于归一化，默认为[0.5, 0.33, 0.33]
        :param std: 标准差，用于归一化，默认为[0.45, 0.5, 0.5]
        :return: 处理后的图像矩阵，形状为(H, W, C)，dtype为int8
        """
        # 1. 填充为正方形
        imageArr = ImageUtils.padAndResize(imageArr, imageSize)
        # 2. 应用PCA
        imageArr = ImageUtils.applyPca(imageArr)
        # 3. 应用基础图像增强
        imageTensor = ImageUtils.imageArrayToTensorWithEnhance(imageArr, mean, std)
        imageArr = ImageUtils.tensorToArray(imageTensor)
        return imageArr

    @staticmethod
    def decodeImageFromBase64(req):
        """
        从 JSON 请求中解码 Base64 图像, 返回np数组
        :param req: Flask request 对象
        :return: numpy array (H, W, C), BGR
        """
        data = req.get_json()
        if not data or 'image' not in data:
            raise ValueError("请求中缺少 'image' 字段")

        image_b64 = data['image']
        try:
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is None:
                raise ValueError("图像解码失败，可能格式不支持或损坏")
            return img
        except Exception as e:
            raise ValueError(f"Base64 解码失败: {str(e)}")


if __name__ == '__main__':
    img = Image.open(r"D:\python项目\智能黑色素瘤识别与分割系统 V0.1\resources\images\test.jpg")
    img_arr = np.array(img)

    pad_arr = ImageUtils.padAndResize(img_arr, 400)
    pca_arr = ImageUtils.applyPca(pad_arr)
    enhance_tensor = ImageUtils.imageArrayToTensorWithEnhance(pca_arr)
    enhance_arr = ImageUtils.tensorToArray(enhance_tensor)
    enhance_img = Image.fromarray(enhance_arr)
    enhance_img.show()
    enhance_arr = ImageUtils.resizeBackToOriginal(enhance_arr, img_arr.shape[:2])
    enhance_img = Image.fromarray(enhance_arr)
    enhance_img.show()
