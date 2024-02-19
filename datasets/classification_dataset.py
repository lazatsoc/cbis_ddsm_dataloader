from datasets.generic_dataset import CBISDDSMGenericDataset


class CBISDDSMClassificationDataset(CBISDDSMGenericDataset):
    def __init__(self, dataframe, download_path, label_field, label_list, masks=False, transform=None, train_image_transform=None, test_image_transform=None):
        super().__init__(dataframe, download_path, masks=masks, transform=transform, train_image_transform=train_image_transform, test_image_transform=test_image_transform)

        self.label_field = label_field
        self.label_list = label_list

    def __getitem__(self, idx):
        image_tensor, item = super().__getitem__(idx)

        label_full = item[self.label_field]
        label = self.label_list.index(label_full)

        return image_tensor, label

    def _get_label_visualize(self, label):
        label = self.label_list[label]
        return label

    @property
    def num_classes(self):
        return len(self.label_list)


