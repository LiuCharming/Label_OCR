# 标签定位与内容识别

批量扫描产品照片，定位标签、读取二维码、校正倾角并识别文字。

默认引擎是 **PP-OCRv4 Mobile**（CPU 轻量模型）。程序优先利用二维码四角作为标签锚点：候选区域必须包含二维码；亮度轮廓不完整时，则由二维码的尺寸和位置估算标签框，从而避免将桌面纹理等背景误判为标签。随后按二维码边缘计算标签倾角，旋正后再进行 OCR。

## 环境要求

- Windows 10/11（已在当前 Windows 环境验证）
- Python 3.13（已验证；建议使用 64 位 Python）
- 网络连接：仅首次运行需要下载 PP-OCR 模型，之后将使用本地缓存
- 无需 GPU；程序默认使用 CPU，检测输入边长限制为 640 以降低资源占用

安装依赖：

```powershell
cd E:\shibie
python -m pip install -r requirements.txt
```

## 使用

```powershell
pip install -r requirements.txt
python label_reader.py .\pic --output .\output
```

如果只需定位铭牌与读取二维码（速度更快，也不下载 OCR 模型）：

```powershell
python label_reader.py .\pic --output .\output --no-ocr
```

也可以直接处理单张图片：

```powershell
python label_reader.py .\pic\902032100_20260914184505_f734.jpg --output .\output
```

## 输出

- `*_label.jpg`：定位出的原始铭牌裁图
- `*_rectified.jpg`：根据二维码四角计算倾角并旋正后的标签图；OCR 优先识别此图
- `*_annotated.jpg`：黄色框标出铭牌位置的原图
- `results.json`：适合程序继续处理的结构化识别结果
- `results.csv`：可直接用 Excel 打开

`label_box` 的格式为 `[x, y, width, height]`，坐标相对于原始图片左上角。`label_angle` 是二维码计算出的原始标签倾角（度）；未检测到二维码四角时为 `null`。

## 验证安装

```powershell
python label_reader.py .\pic\902032100_20260914184505_f734.jpg --output .\smoke_test
```

成功后，终端会显示 `Scanned 1 image(s); located 1 label(s)`。检查 `smoke_test\results.json` 中的 `ocr_text`，应包含产品编码、批次号、数量与 PN 字段。

## 依赖说明

- `paddleocr` + `paddlepaddle`：默认 PP-OCRv4 Mobile 引擎
- `opencv-python`：标签定位、二维码检测和角度校正
- `rapidocr_onnxruntime`：仅在 PaddlePaddle 无法启动时的备用 OCR 引擎
