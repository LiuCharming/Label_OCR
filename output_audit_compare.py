from pathlib import Path
import json
import cv2
root=Path('output_pic_audit_final_20260916')
old=json.loads(Path('output_pic_audit_20260916/results.json').read_text(encoding='utf-8'))
new=json.loads((root/'results.json').read_text(encoding='utf-8'))
for i,(a,b) in enumerate(zip(old,new),1):
    print(i,b['file'], 'lines',len(a['ocr_text'].splitlines()),'->',len(b['ocr_text'].splitlines()),b['ocr_text'].replace('\n',' | '))
    p=root/Path(b['file']).stem/(Path(b['file']).stem+'_rectified.jpg')
    q=Path('output_pic_audit_v3_20260916')/p.relative_to(root)
    x,y=cv2.imread(str(p)),cv2.imread(str(q))
    print('same_v3_image',x.shape==y.shape and bool((x==y).all()))
