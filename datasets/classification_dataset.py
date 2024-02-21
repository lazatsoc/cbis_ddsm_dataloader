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

    def split_train_val(self, val_ratio, shuffle=False, random_state=None):
        df1, df2 = self._split_dataframe(val_ratio, shuffle, random_state)
        val_dataset = CBISDDSMClassificationDataset(df1, self.download_path, self.label_field, self.label_list,
                                                 masks=self.include_masks, transform=self.transform,
                                                 train_image_transform=self._train_image_transforms,
                                                 test_image_transform=self._test_image_transforms)
        val_dataset.test_mode()
        train_dataset = CBISDDSMClassificationDataset(df2, self.download_path, self.label_field, self.label_list,
                                                 masks=self.include_masks, transform=self.transform,
                                                 train_image_transform=self._train_image_transforms,
                                                 test_image_transform=self._test_image_transforms)
        train_dataset.train_mode()
        return val_dataset, train_dataset

    def split_crossval(self, folds, shuffle=False, random_state=None):
        dataframe_pairs = self._split_dataframe_crossval(folds, shuffle, random_state)
        dataset_pairs = []
        for i in range(folds):
            train_dataset = CBISDDSMClassificationDataset(dataframe_pairs[i][0], self.download_path, self.label_field, self.label_list,
                                                    masks=self.include_masks, transform=self.transform,
                                                    train_image_transform=self._train_image_transforms,
                                                    test_image_transform=self._test_image_transforms)
            train_dataset.train_mode()
            val_dataset = CBISDDSMClassificationDataset(dataframe_pairs[i][1], self.download_path, self.label_field, self.label_list,
                                                    masks=self.include_masks, transform=self.transform,
                                                    train_image_transform=self._train_image_transforms,
                                                    test_image_transform=self._test_image_transforms)
            val_dataset.test_mode()
            dataset_pairs.append((train_dataset, val_dataset))
        return dataset_pairs
