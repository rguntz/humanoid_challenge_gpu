from pathlib import Path

import cv2
from PIL import Image

dpi = 300
px_per_mm = dpi / 25.4
tag_px = 7 * round(150 * px_per_mm / 7)      # 5x5 tag = 7x7 cells, keep cells whole
margin_px = round(20 * px_per_mm)

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
img = cv2.aruco.generateImageMarker(d, 0, tag_px)
img = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px,
                         cv2.BORDER_CONSTANT, value=255)
out = Path(__file__).parent / "tag_5x5_id0_150mm.png"
Image.fromarray(img).save(out, dpi=(dpi, dpi))