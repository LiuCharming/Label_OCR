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

OCR 前会在独立副本中用白色遮盖检测到的二维码（包含小幅边缘余量），避免二维码被当成文字。原图、定位、透视矫正和二维码解码保持原始图像。`*_ocr_input.png` 保存遮码后的矫正图；测试页可直接查看。二维码未检测到时无法自动遮盖。

标签定位后会尝试精修白色边界，减少反光外壳进入裁图。测试页新增「疑似反光区域」图，橙色表示大块接近纯白的区域，并提示关闭闪光灯、换角度补拍。该检测是启发式，白底也可能误报；不表示这些区域确定丢失文字。`boundary_refined`、`glare_ratio`、`quality_warning` 记录检测结果，`*_glare.png` 保存标记图。

矫正优先采用二维码四角到正方形的透视变换，适用于二维码与文字在同一平面的标签，可同时消除旋转与斜拍梯形变形。变换不稳定时回退为旋转；四角无法获取时保留原图。`correction_method` 记录实际采用的方法。透视矫正不能恢复原图中缺失或模糊的笔画。

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

## 浏览器测试页

「原图识别」默认关闭，仅 OCR 矫正后的标签；开启后额外 OCR 未矫正的标签裁图，并单独显示结果。所有 OCR 输入仍遮盖二维码。此开关不对整张照片背景进行 OCR。命令行对应 `--original-ocr`。增强开关独立控制额外的增强图 OCR。

启动本地测试页：

```powershell
python app.py
```

浏览器打开 `http://127.0.0.1:5000`。可一次拖入多张图片；每张结果会按同一张图片展示定位框、原始标签裁图、二维码矫正图、二维码内容和 OCR 文字。上传与识别仅在本机完成。

## 依赖说明

### 浅色笔画增强

测试页勾选「图像增强对照」可查看白底黑字增强图及独立 OCR 结果。先确认矫正图中的标签轮廓，遮盖轮廓外背景，再进行保边降噪、纸面光照校正和 Gamma=1.6 温和增强。边界无法确认时跳过增强，通过 `enhancement_note` 提示使用矫正后结果；不会按字符长度删除识别内容。命令行使用 `python label_reader.py .\pic --output .\output --enhance`。
增强增加一次 OCR，默认关闭；结果保留 `ocr_text`，另存 `enhanced_text`、`enhanced_confidence` 和 `*_enhanced.png`。浅色短横线仍可能漏识别，增强不会按编码格式补写字符，请对照图片复核。

- `paddleocr` + `paddlepaddle`：默认 PP-OCRv4 Mobile 引擎
- `opencv-python`：标签定位、二维码检测和角度校正
- `rapidocr_onnxruntime`：仅在 PaddlePaddle 无法启动时的备用 OCR 引擎
