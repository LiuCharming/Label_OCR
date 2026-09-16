import unittest
import cv2
import numpy as np
from label_reader import segment_white_label, rectify_paper


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


if __name__ == '__main__':
    unittest.main()
