import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Workaround found in: https://stackoverflow.com/questions/42462431/oserror-broken-data-stream-when-reading-image-file
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F


class CBISDDSMGenericDataset(Dataset):
    def __init__(self, dataframe, download_path, masks=False, transform=None, train_image_transform=None, test_image_transform=None):
        self.dataframe = dataframe
        self.download_path = download_path
        self.transform = transform
        self.include_masks = masks
        self.current_index = 0
        self.__train_mode = False
        self.__test_mode = False
        self.__train_image_transforms = train_image_transform
        self.__test_image_transforms = test_image_transform


    def __getitem__(self, index):
        item = self.dataframe.iloc[index].to_dict()
        img_path = os.path.join(self.download_path, item['image_path'])
        image = Image.open(img_path)
        max_value = 65536 if image.mode == 'I' else 256
        image_tensor = F.pil_to_tensor(image).float()
        image_tensor /= max_value
        image_tensor_list = [image_tensor]

        if self.include_masks:
            mask_img_path = os.path.join(self.download_path, item['mask_path'])
            mask_image = Image.open(mask_img_path)
            mask_image_tensor = F.pil_to_tensor(mask_image).float()
            mask_image_tensor /= 255
            image_tensor_list.append(mask_image_tensor)

        sample = {'image_tensor_list': image_tensor_list, 'item': item}

        if self.transform is not None:
            sample = self.transform(sample)

        if self.__train_mode and self.__train_image_transforms is not None:
            state = torch.get_rng_state()
            for i in range(len(sample['image_tensor_list'])):
                torch.set_rng_state(state)
                sample['image_tensor_list'][i] = self.__train_image_transforms(sample['image_tensor_list'][i])
        elif self.__test_mode and self.__test_image_transforms is not None:
            state = torch.get_rng_state()
            for i in range(len(sample['image_tensor_list'])):
                torch.set_rng_state(state)
                sample['image_tensor_list'][i] = self.__test_image_transforms(sample['image_tensor_list'][i])

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

    def train_mode(self):
        self.__train_mode = True
        self.__test_mode = False
        return self

    def test_mode(self):
        self.__train_mode = False
        self.__test_mode = True
        return self

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