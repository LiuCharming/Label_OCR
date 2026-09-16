import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from label_reader import segment_white_label, rectify_paper, find_label, scan_image


class WhiteLabelTests(unittest.TestCase):
    def test_large_paper_with_small_bottom_qr(self):
        image = np.full((600, 800, 3), (70, 110, 160), np.uint8)
        cv2.rectangle(image, (100, 80), (700, 520), (220, 220, 220), -1)
        cv2.putText(image, 'TOP TEXT', (140, 140), 0, 1, (0, 0, 0), 2)
        points = np.array([[620,440],[650,440],[650,470],[620,470]], np.float32)
        cv2.fillConvexPoly(image, points.astype(np.int32), (0,0,0))
        mask = segment_white_label(image, points)
        self.assertIsNotNone(mask)
        x,y,w,h = cv2.boundingRect(mask)
        self.assertLessEqual(x, 102)
        self.assertLessEqual(y, 82)
        self.assertGreaterEqual(w, 595)
        self.assertGreaterEqual(h, 435)
        output, interior = rectify_paper(image, mask)
        self.assertEqual(output.shape[:2], interior.shape)

    def test_no_paper(self):
        image = np.full((300,400,3), (60,100,150), np.uint8)
        self.assertIsNone(segment_white_label(image))

    def test_real_orientation_and_clipping_regressions(self):
        names = ['902325600_20260915101556_5b74', '902325600_20260915101614_e071', '902032100_20260914182422_b1d4', '902325600_20260915101629_9c13', '902235600_20260915101241_8abd']
        if not all((Path('pic') / (name+'.jpg')).exists() for name in names):
            self.skipTest('Optional local regression photos not present')
        with TemporaryDirectory() as folder:
            for name in names:
                result = scan_image(Path('pic')/(name+'.jpg'), None, Path(folder))
                self.assertTrue(result.label_found)
                output = cv2.imread(str(Path(folder)/(name+'_rectified.jpg')))
                self.assertGreater(output.shape[1], output.shape[0], name)
                # Resolution/orientation smoke check; text completeness is
                # checked separately against the saved OCR audit and images.
                self.assertGreater(output.shape[0], 200)

    def test_reflection_box_cannot_replace_label(self):
        image = np.zeros((500, 700, 3), np.uint8)
        # Bounding rectangle encloses QR centre, but the bright L does not.
        cv2.rectangle(image, (300,100), (650,130), (255,255,255), -1)
        cv2.rectangle(image, (620,100), (650,350), (255,255,255), -1)
        qr = np.array([[450,180],[510,180],[510,240],[450,240]], np.float32)
        self.assertEqual(find_label(image, qr), (0,0,700,500))


if __name__ == '__main__':
    unittest.main()
