# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, UploadFile, status
from models.OCRModel import *
from models.RestfulModel import *
from paddleocr import PaddleOCR
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
import requests
import os
import tempfile
import numpy as np

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=False,
    lang=OCR_LANGUAGE
)
def _np_to_list(value):
    """仅把需要的 numpy 数组转换为 Python list，其它类型原样返回。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def extract_ocr_data(result):
    """
    从 PaddleOCR predict 返回结构中提取所需字段:
    只返回数组形式: [{ 'input_path': str, 'rec_texts': list[str], 'rec_boxes': list }]
    支持以下几种可能格式:
    1. {'res': {...}}  # 单个结果
    2. [{'res': {...}}, {'res': {...}}]  # 多页结果
    3. 旧格式: list 内元素具备属性 input_path / rec_texts / rec_boxes
    4. 直接是 dict {...}
    """

    debug = os.environ.get("OCR_DEBUG", "0") == "1"

    def _extract_from_dict(d: dict):
        if not isinstance(d, dict):
            return None
        core = d.get('res', d)  # 如果包含 res 用 res，没有就直接用自身
        if not isinstance(core, dict):
            return None
        input_path = core.get('input_path', '')
        rec_texts = core.get('rec_texts')
        if rec_texts is None:
            rec_texts = []
        rec_boxes = core.get('rec_boxes')
        if rec_boxes is None:
            rec_boxes = []
        # 仅当 rec_texts 是 list/tuple 才保留，否则置空，避免出现 numpy 数组被错误当成文本
        rec_texts = list(rec_texts) if isinstance(rec_texts, (list, tuple)) else []
        rec_boxes = _np_to_list(rec_boxes)
        return {
            'input_path': input_path,
            'rec_texts': rec_texts,
            'rec_boxes': rec_boxes
        }

    extracted = []

    # 情况 A: result 是 list
    if isinstance(result, list):
        for item in result:
            data = None
            # dict 情况
            if isinstance(item, dict):
                data = _extract_from_dict(item)
            else:  # 对象属性情况
                input_path = getattr(item, 'input_path', '')
                rec_texts = getattr(item, 'rec_texts', []) or []
                rec_boxes = getattr(item, 'rec_boxes', []) or []
                rec_boxes = _np_to_list(rec_boxes)
                if rec_texts or rec_boxes or input_path:
                    data = {
                        'input_path': input_path,
                        'rec_texts': list(rec_texts) if isinstance(rec_texts, (list, tuple)) else [],
                        'rec_boxes': rec_boxes
                    }
            if data:
                extracted.append(data)
        if extracted:
            return extracted

    # 情况 B: result 是 dict
    if isinstance(result, dict):
        data = _extract_from_dict(result)
        if data:
            return [data]

    # 其它未知情况: 返回空结构，便于前端处理
    if debug:
        print(f"[extract_ocr_data] 未识别的结果类型: {type(result)}")
    return [{'input_path': '', 'rec_texts': [], 'rec_boxes': []}]


@router.get('/predict-by-path', response_model=RestfulModel, summary="识别本地图片")
def predict_by_path(image_path: str):
    result = ocr.predict(input=image_path)
    # 提取关键数据：input_path, rec_texts, rec_boxes
    result_data = extract_ocr_data(result)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result_data, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-base64', response_model=RestfulModel, summary="识别 Base64 数据")
def predict_by_base64(base64model: Base64PostModel):
    img = base64_to_ndarray(base64model.base64_str)
    
    # 保存为临时文件，因为predict方法需要文件路径
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        import cv2
        cv2.imwrite(tmp_file.name, img)
        result = ocr.predict(input=tmp_file.name)
        os.unlink(tmp_file.name)  # 删除临时文件
    
    # 提取关键数据：input_path, rec_texts, rec_boxes
    result_data = extract_ocr_data(result)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result_data, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-file', response_model=RestfulModel, summary="识别上传文件")
async def predict_by_file(file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):  # 支持更多图片格式
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()
            result = ocr.predict(input=tmp_file.name)
            os.unlink(tmp_file.name)  # 删除临时文件
        
        # 提取关键数据：input_path, rec_texts, rec_boxes
        result_data = extract_ocr_data(result)
        restfulModel.data = result_data
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传支持的图片格式 (.jpg, .png, .jpeg, .bmp, .tiff)"
        )
    return restfulModel


@router.get('/predict-by-url', response_model=RestfulModel, summary="识别图片 URL")
async def predict_by_url(imageUrl: str):
    # 直接使用URL进行predict
    result = ocr.predict(input=imageUrl)
    # 提取关键数据：input_path, rec_texts, rec_boxes
    result_data = extract_ocr_data(result)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result_data)
    return restfulModel
