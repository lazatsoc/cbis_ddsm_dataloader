from datasets.generic_dataset import CBISDDSMGenericDataset

class CBISDDSMClassificationDataset(CBISDDSMGenericDataset):
    def __init__(self, dataframe, download_path, label_field, label_list, transform=None):
        super().__init__(dataframe, download_path, transform=transform)

        self.label_field = label_field
        self.label_list = label_list

    def __getitem__(self, idx):
        image_tensor, item = super().__getitem__(idx)

        # abnorm_w = (item['maxx'] - item['minx']) / 2
        # abnorm_x = int(abnorm_w + item['minx'])
        # abnorm_h = (item['maxy'] - item['miny']) / 2
        # abnorm_y = int(abnorm_h + item['miny'])

        label_full = item[self.label_field]
        label = self.label_list.index(label_full)

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)

        return image_tensor, label

    def _get_label_visualize(self, label):
        label = self.label_list[label]
        return label


