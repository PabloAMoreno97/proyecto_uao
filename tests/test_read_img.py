import cv2
import numpy as np

from src.data.read_img import read_jpg_file


def test_read_jpg_file(tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    file = tmp_path / "dummy.jpg"
    cv2.imwrite(str(file), img)

    arr, img_pil = read_jpg_file(str(file))
    assert arr.shape == (100, 100, 3) or len(arr.shape) == 2
    assert img_pil is not None
