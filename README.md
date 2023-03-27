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
With `CBISDDSMDatasetFactory` either one or both of them can be accessed via the `.train()` and
`.test()` functions:

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

#### B. Random patch transform
By using the option 
```python
.lesion_patches_random(shape = (1024, 1024), min_overlap=0.9)
```
the factory will provide random patches of size `shape`, sampled on random locations around the lesion. 
The `min_overlap` parameter specifies the minimum percentage of overlap that the patch should have with the lesion.
