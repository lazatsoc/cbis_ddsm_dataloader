from ddsm_dataset_factory import CBISDDSMDatasetFactory

dataset = CBISDDSMDatasetFactory('./config.json',
                                                    include_train_set=True,
                                                    include_test_set=True,
                                                    include_calcifications=False,
                                                    include_masses=True) \
        .drop_attributes("assessment", "breast_density", "subtlety") \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}) \
        .show_counts() \
        .lesion_patches_centered() \
        .create_classification('pathology')
dataset[0].visualize()