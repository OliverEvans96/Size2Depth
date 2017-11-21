from data import get_rgbd
import skimage.io as io
import numpy as np

ind = 233
(rgb, depth) = get_rgbd(ind)

(H, W) = depth.shape
label = []
for div in (4,10,20):
    rgb_img = rgb.astype('float32')
    for row in range(H / div, H, H / div):
        rgb_img[row, :, :] = 255
    for col in range(W / div, W, W / div):
        rgb_img[:, col, :] = 255
    io.imshow(rgb_img)
    io.show()

    size = np.zeros(div, div)
    for r in range(div):
        for c in range(div):
            size[r][c] = input()
    label.append(size)

def get_Label():
    return label