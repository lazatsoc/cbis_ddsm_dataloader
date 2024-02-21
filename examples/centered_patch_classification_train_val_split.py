from ddsm_dataset_factory import CBISDDSMDatasetFactory
from torchvision import transforms

dataset = CBISDDSMDatasetFactory('./config.json') \
        .drop_attributes("assessment", "breast_density", "subtlety") \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}) \
        .show_counts() \
        .lesion_patches_centered() \
        .add_image_transforms([transforms.Resize(512)]) \
        .add_image_transforms([transforms.RandomAffine(degrees=180, scale=(0.7, 1.5)),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip()], for_val=False) \
        .add_image_transforms([transforms.Lambda(lambda x: x.repeat(3, 1, 1))]) \
        .create_classification('pathology', mask_input=True)

print(len(dataset))
train_set, val_set = dataset.split_train_val(0.2, shuffle=True)
print(len(train_set), len(val_set))
train_set.visualize()