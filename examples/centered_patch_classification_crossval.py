from ddsm_dataset_factory import CBISDDSMDatasetFactory
from torchvision import transforms

fold1, fold2, fold3, fold4, fold5 = CBISDDSMDatasetFactory('./config.json') \
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


fold1[0].visualize()