import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import math


try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class AutoXYDataset(data.Dataset):
    """
    Read X (Low Quality, here is X) and Y image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.X_paths, self.Y_paths = None, None
        self.X_env, self.Y_env = None, None  # environment for lmdb
        self.X_size, self.Y_size = opt["X_size"], opt["Y_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.X_paths, self.X_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_X"]
            )
            self.Y_paths, self.Y_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_Y"]
            )
        elif opt["data_type"] == "img":
            self.X_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_X"]
            )  # X list
            self.Y_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_Y"]
            )  # Y list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.Y_paths, "Error: Y paths are empty."
        if self.X_paths and self.Y_paths:
            assert len(self.X_paths) == len(
                self.Y_paths
            ), "Y and X datasets have different number of images - {}, {}.".format(
                len(self.X_paths), len(self.Y_paths)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.Y_env = lmdb.open(
            self.opt["dataroot_Y"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.X_env = lmdb.open(
            self.opt["dataroot_X"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.Y_env is None) or (self.X_env is None):
                self._init_lmdb()

        Y_path, X_path = None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        Y_size = self.opt["Y_size"]
        X_size = self.opt["X_size"]

        # get Y image
        Y_path = self.Y_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.Y_sizes[index].split("_")]
        else:
            resolution = None
        img_Y = util.read_img(
            self.Y_env, Y_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]
        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_Y = util.modcrop(img_Y, scale)

        # get X image
        if self.X_paths:  # X exist
            X_path = self.X_paths[index]
            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.X_sizes[index].split("_")]
            else:
                resolution = None
            img_X = util.read_img(self.X_env, X_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt["phase"] == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_Y.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, Y_size)
                W_s = _mod(W_s, random_scale, scale, Y_size)
                img_Y = cv2.resize(
                    np.copy(img_Y), (W_s, H_s), interpolation=cv2.INTER_LINEAR
                )
                # force to 3 channels
                if img_Y.ndim == 2:
                    img_Y = cv2.cvtColor(img_Y, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_Y.shape
            # using matlab imresize
            img_X = util.imresize(img_Y, 1 / scale, True)
            if img_X.ndim == 2:
                img_X = np.expand_dims(img_X, axis=2)
                



        if self.opt["phase"] == "train":

            H_Y, W_Y, _ = img_Y.shape
            H_X, W_X, _ = img_X.shape

            if H_Y < Y_size or W_Y < Y_size:
                scale_Y = max(Y_size / H_Y, Y_size / W_Y)
                new_H_Y = math.ceil(H_Y * scale_Y)
                new_W_Y = math.ceil(W_Y * scale_Y)
                img_Y = cv2.resize(img_Y, (new_W_Y, new_H_Y), interpolation=cv2.INTER_LINEAR)
        
        
                scale_X = scale_Y
                new_H_X = math.ceil(H_X * scale_X)
                new_W_X = math.ceil(W_X * scale_X)
                img_X = cv2.resize(img_X, (new_W_X, new_H_X), interpolation=cv2.INTER_LINEAR)

            if img_Y.ndim == 2:
                img_Y = np.expand_dims(img_Y, axis=2)
            if img_X.ndim == 2:
                img_X = np.expand_dims(img_X, axis=2)

            H_X, W_X, _ = img_X.shape
            H_Y, W_Y, _ = img_Y.shape
            
            assert X_size <= H_X and X_size <= W_X, f"X image too small: {(H_X, W_X)} < {X_size}"
            assert Y_size <= H_Y and Y_size <= W_Y, f"Y image too small: {(H_Y, W_Y)} < {Y_size}"            
            
            
            # randomly crop
            rnd_h = random.randint(0, max(0, H_X - X_size))
            rnd_w = random.randint(0, max(0, W_X - X_size))
            img_X = img_X[rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :]
            rnd_h_Y, rnd_w_Y = int(rnd_h), int(rnd_w)
            img_Y = img_Y[rnd_h_Y : rnd_h_Y + Y_size, rnd_w_Y : rnd_w_Y + Y_size, :]
#            print('ooooooooooo')
#            print(img_Y.shape)
#            print(img_X.shape)
#            print('ooooooooooo')
            # augmentation - flip, rotate
            img_X, img_Y = util.augment(
                [img_X, img_Y],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
                self.opt["use_swap"],
            )
        elif X_size is not None:
            H, W, C = img_X.shape
            assert X_size == Y_size // scale, "Y size does not match X size"

            if X_size < H and X_size < W:
                # center crop
                rnd_h = H // 2 - X_size//2
                rnd_w = W // 2 - X_size//2
                img_X = img_X[rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :]
                rnd_h_Y, rnd_w_Y = int(rnd_h * scale), int(rnd_w * scale)
                img_Y = img_Y[
                    rnd_h_Y : rnd_h_Y + Y_size, rnd_w_Y : rnd_w_Y + Y_size, :
                ]

        # change color space if necessary
        if self.opt["color"]:
            H, W, C = img_X.shape
            img_X = util.channel_convert(C, self.opt["color"], [img_X])[
                0
            ]  # TODO during val no definition
            img_Y = util.channel_convert(img_Y.shape[2], self.opt["color"], [img_Y])[
                0
            ]
        
        if img_X.shape[2] == 1:
            img_X = np.dstack((img_X, img_X, img_X))

        if img_Y.shape[2] == 1:
            img_Y = np.dstack((img_Y, img_Y, img_Y))

        
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_Y.shape[2] == 3:
            #print('ooooooooooo')
            #print(img_Y.shape)
            img_Y = img_Y[:, :, [2, 1, 0]]
            img_X = img_X[:, :, [2, 1, 0]]
        img_Y = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_Y, (2, 0, 1)))
        ).float()
        img_X = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_X, (2, 0, 1)))
        ).float()

        if X_path is None:
            X_path = Y_path

        return {"X": img_X, "Y": img_Y, "X_path": X_path, "Y_path": Y_path}

    def __len__(self):
        return len(self.Y_paths)
