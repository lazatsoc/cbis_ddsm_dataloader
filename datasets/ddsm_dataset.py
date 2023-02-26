import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
import numpy as np
import torch

class CBISDDSMCenteredPatchesDataset(Dataset):
    def __init__(self, dataframe, download_path, label_field, label_list, orig_patch_size=(1024,1024), bits=8):
        self.dataframe = dataframe
        self.download_path = download_path
        self.bits = bits
        self.orig_patch_size = orig_patch_size
        self.label_field = label_field
        self.label_list = label_list

        self.pil2tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img_path = os.path.join(self.download_path, item['image_path'])
        image = Image.open(img_path)

        cx = item['cx']
        cy = item['cy']

        minx_naive = int(cx - self.orig_patch_size[0] / 2)
        minx = max((0, minx_naive))
        dx1 = minx_naive - minx

        maxx_naive = int(cx + self.orig_patch_size[0] / 2)
        maxx = min((maxx_naive, image.size[0]))
        dx2 = maxx - maxx_naive

        if dx1 != 0 and dx2 != 0:
            print('Warning: patch size bigger than image x-dimension. Please select a smaller patch size.')
        else:
            minx -= dx2
            maxx -= dx1

        miny_naive = int(cy - self.orig_patch_size[1] / 2)
        miny = max((0, miny_naive))
        dy1 = miny_naive - miny

        maxy_naive = int(cy + self.orig_patch_size[1] / 2)
        maxy = min((maxy_naive, image.size[1]))
        dy2 = maxy - maxy_naive

        if dy1 != 0 and dy2 != 0:
            print('Warning: patch size bigger than image y-dimension. Please select a smaller patch size.')
        else:
            miny -= dy2
            maxy -= dy1

        image = image.crop((minx, miny, maxx, maxy))

        # abnorm_w = (item['maxx'] - item['minx']) / 2
        # abnorm_x = int(abnorm_w + item['minx'])
        # abnorm_h = (item['maxy'] - item['miny']) / 2
        # abnorm_y = int(abnorm_h + item['miny'])

        image_arr = np.array(image).astype(np.float32)
        image_arr /= 65535
        image_tensor = torch.from_numpy(image_arr)

        label_full = item[self.label_field]
        label = self.label_list.index(label_full)

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)

        return image_tensor, label

    def visualize(self):
        for i in range(len(self)):
            image, label = self[i]
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(self.label_list[label])
            plt.waitforbuttonpress()
