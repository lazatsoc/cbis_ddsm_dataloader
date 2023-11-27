from ddsm_dataset_factory import CBISDDSMDatasetFactory
from torchvision import transforms

datasets = CBISDDSMDatasetFactory('./config.json') \
        .train_test_merge() \
        .add_masses() \
        .drop_attributes("assessment", "breast_density", "subtlety") \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}) \
        .show_counts() \
        .lesion_patches_centered() \
        .add_image_transforms([transforms.Resize(512)]) \
        .add_image_transforms([transforms.RandomAffine(degrees=180, scale=(0.7, 1.5)),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip()], for_val=False) \
        .add_image_transforms([transforms.Lambda(lambda x: x.repeat(3, 1, 1))]) \
        .split_cross_validation(5) \
        .create_classification('pathology', mask_input=True)


datasets[0][0].visualize()