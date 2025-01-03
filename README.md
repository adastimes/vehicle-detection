# vehicle-detection

Create virtual environment:
    python -m venv .venv

Load requierments packages from rtdeter_pytorch:
    pip install -r requirements.txt

1. Download the kitti dataset and annotation (12GB + 5MB) for objects : https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark
2. [not mandatory]Place the images in a training folder, and a test folder (they come by default in a subdir from the respective camera ).
3. [not mandatory]Place the label txt file in a folder called labels; default they come in a subfolder labels_2
4. [not mandatory]Create a directory annotations where the coco json files will be placed.
5. Adjust paths to this directories in kitti2coco.py and run. It will generate two json files, one for training and one for validation
6. Follow the guidance from rtdeter and run.. Adjust config files to data location, json files.