"""One-off full-folder audit; does not change production processing."""
from pathlib import Path
from dataclasses import asdict
import argparse, json, time, html
import cv2
import numpy as np
from label_reader import iter_images, build_ocr, scan_image

parser = argparse.ArgumentParser(description='Grouped visual/OCR regression audit')
parser.add_argument('--source', type=Path, default=Path('pic'))
parser.add_argument('--output', type=Path, default=Path('output_pic_audit_v3_20260916'))
args = parser.parse_args()
root = args.output
root.mkdir(exist_ok=True)
ocr = build_ocr('ch')
records = []
tiles = []
for i, source in enumerate(iter_images(args.source), 1):
    folder = root / source.stem
    folder.mkdir(exist_ok=True)
    start = time.perf_counter()
    try:
        result = asdict(scan_image(source, ocr, folder, enhance=True))
    except Exception as exc:
        result = {'file': source.name, 'error': repr(exc)}
    result['seconds'] = round(time.perf_counter()-start, 2)
    records.append(result)
    (root/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    tile = np.full((470,1500,3),245,np.uint8)
    cv2.putText(tile, f'{i:02} {source.name}', (12,25), 0,.65,(0,0,0),1)
    paths = [folder/f'{source.stem}_annotated.jpg', folder/f'{source.stem}_rectified.jpg', folder/f'{source.stem}_enhanced.png']
    for j,path in enumerate(paths):
        im=cv2.imread(str(path)) if path.exists() else None
        if im is None: continue
        h,w=im.shape[:2]; factor=min(490/w,425/h)
        im=cv2.resize(im,(round(w*factor),round(h*factor)))
        y=38+(425-im.shape[0])//2; x=j*500+(500-im.shape[1])//2
        tile[y:y+im.shape[0],x:x+im.shape[1]]=im
    tiles.append(tile)
    if len(tiles)==4:
        cv2.imwrite(str(root/f'contact_{i//4:02}.jpg'),np.vstack(tiles)); tiles=[]
    print(f'{i:02} {source.name} {result}',flush=True)
if tiles: cv2.imwrite(str(root/f'contact_{(len(records)+3)//4:02}.jpg'),np.vstack(tiles))
cards=[]
for i,r in enumerate(records,1):
    stem=Path(r['file']).stem
    images=''.join(f'<a href="{stem}/{stem}_{s}"><img src="{stem}/{stem}_{s}"></a>' for s in ['annotated.jpg','rectified.jpg','enhanced.png'] if (root/stem/f'{stem}_{s}').exists())
    cards.append(f'<article><h2>{i:02} {html.escape(r["file"])}</h2>{images}<pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre></article>')
(root/'index.html').write_text('<meta charset="utf-8"><title>逐张测试</title><style>body{font-family:Arial;background:#eee}article{background:white;padding:16px;margin:16px}img{width:32%;height:330px;object-fit:contain}pre{white-space:pre-wrap}</style>'+''.join(cards),encoding='utf-8')
