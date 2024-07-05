# ORCGT: Ollivier-Ricci Curvature-based Graph Model for Lung STAS Prediction
![Overview](/Pics/overview-5.pdf)

## Installation
Clone the repo:
```bash
git clone https://github.com/zhengwang9/STAS.git && cd STAS
```
Create a conda environment and activate it:
```bash
conda create -n env python=3.9
conda activate env
pip install -r requirements.txt
```

## Major Tumor Margin Extraction
We employ a pretrained HoVerNet to classify tumor patches based on their cell count. Then, we derive the mask for the major tumor region using UNet with tumor dentisy map.
```bash
#classify tumor patches
python ./hv_res_post-process/choose_tumor_patch.py
#make tumor density map
python ./hv_res_post-process/tumor_density.py
#choose Ring patches
python ./hv_res_post-process/choose_ring.py
```

## Training

You also can download the processed graph data [here](https://cloud.189.cn/t/NziQRbUrAJju). The access code is: dei3

## Training
First, setting the data splits and hyperparameters in the file ***train.py***. Then, experiments can be run using the following command-line:
```bash
cd train
python train_<experiments>.py
```
The trained model will be saved in the folder ***SavedModels***. 





