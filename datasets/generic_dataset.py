import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt


class CBISDDSMGenericDataset(Dataset):
    def __init__(self, dataframe, download_path, masks=False, transform=None, image_tranform=None):
        self.dataframe = dataframe
        self.download_path = download_path
        self.transform = transform
        self.image_transforms = image_tranform
        self.include_masks = masks
        self.current_index = 0

    def __getitem__(self, index):
        item = self.dataframe.iloc[index].to_dict()
        img_path = os.path.join(self.download_path, item['image_path'])
        image = Image.open(img_path)

        image_arr = np.array(image).astype(np.float32)
        image_arr /= 65535
        # image_arr = np.expand_dims(image_arr, 0)
        image_tensor = torch.from_numpy(image_arr)

        image_tensor_list = [image_tensor]

        if self.include_masks:
            mask_img_path = os.path.join(self.download_path, item['mask_path'])
            mask_image = Image.open(mask_img_path)

            mask_image_arr = np.array(mask_image).astype(np.float32)
            mask_image_arr /= 255
            # mask_image_arr = np.expand_dims(mask_image_arr, 0)
            mask_image_tensor = torch.from_numpy(mask_image_arr)

            image_tensor_list.append(mask_image_tensor)

        sample = {'image_tensor_list': image_tensor_list, 'item': item}

        if self.transform is not None:
            sample = self.transform(sample)

        for i in range(len(sample['image_tensor_list'])):
            sample['image_tensor_list'][i] = torch.unsqueeze(sample['image_tensor_list'][i], 0)

        if self.image_transforms is not None:
            state = torch.get_rng_state()
            for i in range(len(sample['image_tensor_list'])):
                torch.set_rng_state(state)
                sample['image_tensor_list'][i] = self.image_transforms(sample['image_tensor_list'][i])

        return sample['image_tensor_list'], sample['item']

    def __len__(self):
        return len(self.dataframe.index)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self):
            x = self[self.current_index]
            self.current_index += 1
            return x
        else:
            raise StopIteration

    def _get_img_visualize(self, image):
        return image

    def _get_label_visualize(self, item):
        return f'{item["patient_id"]}_{item["left_right"]}_{item["view"]}'

    def visualize(self):
        if self.include_masks:
            figure = plt.figure(figsize=(1, 2))
        else:
            figure = plt.figure()

        def plot(e):
            plt.clf()
            image_list, item = next(self)

            image = image_list[0].transpose(0, 2)

            if self.include_masks:
                figure.add_subplot(1, 2, 1)

                mask = image_list[1].transpose(0, 2)

                plt.imshow(self._get_img_visualize(image))
                plt.title(self._get_label_visualize(item), backgroundcolor='white')
                figure.add_subplot(1, 2, 2)
                plt.imshow(self._get_img_visualize(mask), cmap='gray')
            else:
                plt.imshow(self._get_img_visualize(image), cmap='gray')
                plt.title(self._get_label_visualize(item), backgroundcolor='white')
            plt.draw()

        figure.canvas.mpl_connect('key_press_event', plot)
        plt.show()