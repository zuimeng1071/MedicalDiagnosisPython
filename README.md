# Medical Diagnosis Python

## 项目描述

这是一个基于深度学习的医疗诊断Python应用，主要功能包括：

- **图像分割**：使用U-Net模型对医疗图像进行分割。
- **文档分割**：使用NLP模型对文档进行分割处理。
  - 来源：https://modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/feedback/issueDetail/12631

该项目使用Flask框架提供REST API接口，支持图像上传和分割结果返回。


## 项目结构

- `run.py`：主应用程序文件，包含Flask服务器和API定义
- `models/`：深度学习模型定义
  - `DASPP_ChannelAtte_UNet.py`：改进的U-Net模型
  - `Unet.py`：标准U-Net模型
- `resources/`：资源文件
  - `images/`：示例图像
  - `modelWeight/segment.pth`：图像分割模型权重
  - `nlpDocumentSegmentation/`：NLP文档分割模型
- `tools/`：工具类
  - `ImageUtils.py`：图像处理工具
  - `Segment.py`：分割工具

## 依赖项

- Python 3.11
- PyTorch 2.6.0+cu118
- TorchVision 0.21.0+cu118
- Torchaudio 2.6.0+cu118
- OpenCV-Python 4.11.0.86
- Flask 3.1.2
- ModelScope 1.33.0
- Transformers 4.48.3
- HuggingFace-Hub 0.25.2


## 模型及模型权重文件：
- 文本分割模型：
  - modelscope download --model iic/nlp_bert_document-segmentation_chinese-base README.md --local_dir ./dir
- 图像分割模型：
  - 链接: https://pan.baidu.com/s/1b_QgkPLH-smhjTBTWJqzxw?pwd=j6n5 提取码: j6n5 
