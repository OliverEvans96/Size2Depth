import skimage.io as io
import numpy as np
from predefined_labels import get_depth

# Function to get new label from user
def get_label(rgb):
    (H, W, _) = rgb.shape
    depth_maps = []
    sizes = [7]
    for div in sizes:    #
        rgb_img = rgb.astype('float32')
        for row in range(int(round(H / div)), H, int(round(H / div))):
            rgb_img[row, :, :] = 255
        for col in range(int(round(W / div)), W, int(round(W / div))):
            rgb_img[:, col, :] = 255
        io.imshow(rgb_img / 255)
        io.show()

        size = np.zeros([div, div])
        for r in range(div):
            for c in range(div):
                print 'input size label for block (%d, %d)'% (r, c)
                size[r][c] = input()
        depth_maps.append(size)
    return depth_maps[0]


# Function to get predefined label
def get_dense_label(cycle):
    (H, W) = (63, 84)
    sizes = [7]
    depth = get_depth()[cycle]
    depth_maps = []
    for div in sizes:
        stepr = int(round(H/div))
        stepc = int(round(W/div))
        dense_depth = np.zeros([H, W])
        for r in range(H):
            for c in range(W):
                pr = int(r/stepr)
                pc = int(c/stepc)
                dense_depth[r][c] = depth[pr][pc]
        depth_maps.append(dense_depth)
    return depth_maps[0]