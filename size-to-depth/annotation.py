import skimage.io as io
from data import DataLoader
import numpy as np


class Annotator:
    def __init__(self):
        self.data = DataLoader(n=32)

    def demo(self, index):
        rgb, depth = self.data[index]

        H, W = depth.shape
        print(rgb.shape)

        print(H, W)

        label = []
        for div in (4,10,20):
            tmp = rgb.copy()
            for row in range(H // div, H, H // div):
                tmp[row, :, :] = 255
            for col in range(W // div, W, W // div):
                tmp[:, col, :] = 255

            print("input %d x %d " % (div, div))
            io.imshow(tmp)
            io.show()

            # size = np.zeros((div, div))
            # for r in range(div):
            #     size[r] = input()
            # label.append(size)

        print(label)
        return label

if __name__ == '__main__':
    annotator = Annotator()
    annotator.demo(0)

