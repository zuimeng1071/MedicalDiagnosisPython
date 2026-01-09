import logging
from base64 import b64encode

import time

import cv2
import torch
from flask import Flask, request, jsonify

from tools.ImageUtils import ImageUtils
from tools.Segment import Segment

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

app = Flask(__name__)
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentModel = Segment.load_model('DASPPChannelAtteUnet',
                                  image_size=(400, 400),
                                  isLoadWeight=True,
                                  weight_path='resources/modelWeight/segment.pth')
# 全局加载模型（启动时加载一次）
logger.info("正在加载图像分割模型...")
segmentModel = segmentModel.to(device)
logger.info("图像分割模型加载完成")

logger.info("正在加载文档分割模型...")
text_segment_pipeline = pipeline(
    task=Tasks.document_segmentation,
    model=r'resources/nlpDocumentSegmentation'
)
logger.info("文档分割模型加载完成")


# 图像分割
@app.route('/segment', methods=['POST'])
def segmentImage():
    start_time = time.time()
    # 解码图像
    img_arr = ImageUtils.decodeImageFromBase64(request)
    tensor = ImageUtils.ImagePreProcessing(img_arr, imageSize=400)
    tensor = tensor.to(device)
    # 调用模型进行图像分割
    out_arr, _ = Segment.predict(segmentModel, tensor)
    # 输出尺寸恢复处理
    out_arr = ImageUtils.resizeBackToOriginal(out_arr, img_arr.shape[:2])
    # 合并原始图像与分割结果
    merged_img = ImageUtils.mergeMaskAndOriginal(img_arr, out_arr)

    # 将处理后的图像编码为JPEG格式
    _, img_encoded_merged = cv2.imencode('.jpg', merged_img)
    _, img_encoded_binary = cv2.imencode('.jpg', out_arr)

    # 将图像数据转换为base64字符串
    img_base64_merged = b64encode(img_encoded_merged).decode('utf-8')
    img_base64_binary = b64encode(img_encoded_binary).decode('utf-8')
    logging.info(f'分割任务处理完成, 耗时{time.time() - start_time}')

    # 返回包含两张图像的JSON响应
    response = {
        "mergedImage": img_base64_merged,
        "binaryMask": img_base64_binary
    }

    return jsonify(response), 200


@app.route('/text-segment', methods=['POST'])
def text_segment():
    start_time = time.time()

    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "请求体必须包含 'text' 字段"}), 400

        raw_text = data['text']
        if not isinstance(raw_text, str) or not raw_text.strip():
            return jsonify({"error": "'text' 必须是非空字符串"}), 400

        logger.info(f"收到文本分割请求，长度: {len(raw_text)} 字符")

        # 调用模型进行文档分割
        result = text_segment_pipeline(documents=raw_text)
        segmented_text = result['text']  # 模型返回已按段落分割的文本，段落间用 \n 分隔

        # 可选：返回段落列表（更结构化）
        paragraphs = [p.strip() for p in segmented_text.split('\n') if p.strip()]

        response = {
            "segmentedText": segmented_text,  # 原始分割结果（带 \n）
            "paragraphs": paragraphs,  # 段落列表（方便前端处理）
            "paragraphCount": len(paragraphs),
            "processingTimeMs": int((time.time() - start_time) * 1000)
        }

        logger.info(f"文本分割完成，共 {len(paragraphs)} 段，耗时 {(time.time() - start_time):.2f} 秒")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"文本分割发生异常: {str(e)}", exc_info=True)
        return jsonify({"error": "服务器内部错误", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
