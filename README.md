# CBIS DDSM Dataloader

This repository facilitates the pre-processing of the CBIS-DDSM mammographic database.
It involves downloading the whole database, converting the DICOM images to PNG and parsing the database files.
Finally, a custom PyTorch Dataset and Dataloader can be created in a versatile way, covering whole mammograms, patches
and segmentation masks of the lesions (see examples below).

## Setup
First, install the project requirements using

```shell
pip install -r requirements.txt
```
Next, the `config.json` file should be edited to specify the `download_path` variable.
By default, the database will be downloaded in the root directory, in a new folder `CBIS-DDSM`. Alternatively, edit
the file specifying a different dir:
```json lines
{
  "download_path": "< path to download>",
  ...
}
```
Next, run `setup.py`. The command-line arguments are supported:
```shell
optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Path to the configuration file. Default=config.json
  -d                    If used, dcm file will be deleted during conversion, to free up space.However, if download runs again it will need to download the whole dataset again.
```
The `setup.py` script will download the database to the provided path, convert 
the images to PNG format and pre-process the database csv files. Note that separate codes for each one of these 
processes are provided in the `utils` folder.

## Creating a dataset
Datasets are created using the class `CBISDDSMDatasetFactory` that provides a verstatile way to filter lesions,
manage their attributes and apply transformations on the corresponding images. A detailed description of the factory
functions is given below. Additionally, the folder `examples` provides common cases of dataset creation. Keep in mind that a different copy of `config.json` is
provided in this folder, that should point to the same `download_path` with the original setup config.

### Training / Testing subsets
The CBIS-DDSM database provides two subsets for training and testing purposes, respectively. 
With `CBISDDSMDatasetFactory` either one can be accessed via the `.train()` and
`.test()` functions or both of them merged via the `.train_test_merge()` function:

```python
dataset = CBISDDSMDatasetFactory('./config.json').train()
```

### Mass / Micro-calcification subsets
CBIS-DDSM provides two distinct subsets, concerning lesions of type `mass` or `calcification`. 
With `CBISDDSMDatasetFactory` either one or both of them can be accessed via the `.add_masses()` and 
`.add_calcifications()` functions:

```python
dataset = CBISDDSMDatasetFactory('./config.json').train().add_masses()
```

### Attribute manipulation
In CBIS-DDSM, a broad set of attributes is provided for each lesion. The `CBISDDSMDatasetFactory` provides the function
`.map_attribute_value()` to change a specific value into another one. For example, it is a common case for the 
`pathology` label `BENIGN_WITHOUT_CALLBACK` to be changed to `BENIGN`. This can be achieved with the following code:
```python
dataset = CBISDDSMDatasetFactory('./config.json') \
        .train() \
        .add_masses() \
        .map_attribute_value('pathology', {'BENIGN_WITHOUT_CALLBACK': 'BENIGN'})
```
Additionally, attributes that are not relevant can be dropped via the `.drop_attributes()` method:

```python
dataset = CBISDDSMDatasetFactory('./config.json') \
        .train() \
        .add_masses() \
        .drop_attributes("assessment", "breast_density", "subtlety")
```

### Patch transforms

By default, the examples above will provide whole mammogram images. However, processing lesion patches is a common case
for mammographic CAD systems. `CBISDDSMDatasetFactory` provides two types of patch transforms:

#### A. Centered patch transform
By using the option 
```python
.lesion_patches_centered(shape = (1024, 1024))
``` 
the factory will provide patches that are centered around each lesion.
The `shape` parameter specifies the dimensions of the patch in pixels. 
The default size is set to `(1024, 1024)`, which is sufficient for all the masses in the dataset.

```python
dataset = CBISDDSMDatasetFactory('./config.json') \
        .train() \
        .add_masses() \
        .lesion_patches_centered()
```
Please note that there are some cases where the mass is located near the boundary of the mammogram image. 
In these cases the patch is adjusted (translated) to contain the mass even if it is not centered.
An example of this option is given in `examples/centered_patch_classification_dataset.py`.
#### B. Random patch transform
By using the option 
```python
.lesion_patches_random(shape = (1024, 1024), min_overlap=0.9)
```
the factory will provide random patches of size `shape`, sampled on random locations around the lesion. 
The `min_overlap` parameter specifies the minimum percentage of overlap that the patch should have with the lesion.
An example of this option is given in `examples/random_patch_classification_dataset.py`.
### Image transforms
`CBISDDSMDatasetFactory` supports the application of PyTorch image transforms on the CBIS-DDSM samples,
both whole images and patches. This is achieved via the function
```python
.add_image_transforms(transform_list, for_train = True, for_val = True)
```
that accepts a list of transforms. The parameters `for_train` and `for_val` constrain the application of the 
transform to a specific mode (training mode or validation mode). In this way, the preprocessing transforms can be applied
to all the samples, but the augmentation transforms can be applied only for training.
After the dataset creation, the functions `.train_mode()` and `test_mode()` activate the corresponding configuration.
An example of this option is given in `examples/centered_patch_classification_train_val_split.py`.
### Splitting
`CBISDDSMDatasetFactory` provides two options for splitting the dataset for training and validation 
purposed. 

#### A. Train-val split
By using the option
```python
.split_train_val(validation_percentage=0.2)
```
the factory will return a tuple with two distinct datasets, one for training and one for testing. 
The parameter `validation_percentage` specifies the ratio that will be held out for validation.
An example of this option is given in `examples/centered_patch_classification_train_val_split.py`
#### B. Cross-validation
By using the option
```python
.split_cross_validation(k_folds=5)
```
the factory will return a tuple with `k_fold` splits of the dataset in training/validation. For each split, the training
dataset will contain a ratio of `(k_fold - 1)/k_fold` of the total samples while the validation set will
contain `1/k_fold` of the total samples. The partitioning is performed in a mutually exclusive fashion, i.e. 
a sample is used exactly `k_fold` times for validation. An example of this option is given in 
`examples/centered_patch_classification_crossval.py`.