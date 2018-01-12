import numpy as np
import h5py
import pickle

class DataLoader:
    def __init__(self, n=30):
        if n == -1: # all
            self.data = self._read_mat("nyu_depth_v2_labeled.mat")
            self.data["images"] = self.data["images"].transpose(0, 3, 2, 1)
            self.data["depths"] = self.data["depths"].transpose(0, 2, 1)
        else:
            with open("sample-%d.pickle" % n, "rb") as fin:
                self.data = pickle.load(fin)

        self.rgb = self.data["images"]
        self.depth = self.data["depths"]

    def _read_mat(self, filename, want_list=['images', 'depths']):
        ret = {}
        data = h5py.File(filename)
        for k, v in data.items():
            if k in want_list:
                print(k, v)
                ret[k] = np.array(v)
        return ret

    def __getitem__(self, index):
        return (self.rgb[index], self.depth[index])

    def save_small_sample(self, n=30):
        indexes = np.random.choice(len(self.rgb), n)

        save = {"images": self.rgb[indexes],
                "depths": self.depth[indexes]}

        with open("sample-%d.pickle" % n, "wb") as fout:
            pickle.dump(save, fout)

if __name__ == '__main__':
    print("make small dataset...")
    loader = DataLoader(n=-1)  # -1 for full dataset
    loader.save_small_sample()

