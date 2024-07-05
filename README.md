# [MICCAI 2024] ORCGT: Ollivier-Ricci Curvature-based Graph Model for Lung STAS Prediction
Min Cen*, Zheng Wang*, Zhenfeng Zhuang, Hong Zhang, Dan Su, Zhen Bao, Weiwei Wei, Baptiste Magnier, Lequan Yu, Liansheng Wang†
![Overview](/Pics/ORCGT.png)

## Installation
- Clone the repo:
```bash
- git clone https://github.com/zhengwang9/STAS.git && cd STAS
```
- Create a conda environment and activate it:
```bash
conda create -n env python=3.9
conda activate env
pip install -r requirements.txt
```
## Image Preprocession and Feature Extraction

- We used [CLAM](https://github.com/mahmoodlab/CLAM) to split the slides and extract featurers of patches by [Ctranspath](https://github.com/Xiyue-Wang/TransPath)   

## Major Tumor Margin Extraction
- We employ a pretrained [HoVerNet](https://github.com/vqdang/hover_net) to classify tumor patches based on their cell count.
- Then, we construct the tumor density map by *tumor_density.py*. After that, we derive the mask for the major tumor region using UNet by tumor density map.
- Finally we select patches in ring of major tumor margin by *choose_ring.py*.
```bash
# classify tumor patches
python ./hv_res_post-process/choose_tumor_patch.py
# construct tumor density map
python ./hv_res_post-process/tumor_density.py
# select Ring patches
python ./hv_res_post-process/choose_ring.py
```

## Graph Construction

We construct the graph with curvature by two steps: 
- extract feature of patches (nodes) in major tumor margin
- construct graph.

```bash
# extract feats in ring.
python ./graph_construction/extract_huandai_feats.py
# construct graph
python ./graph_construction/ToPyG_curva.py
```

## Training

```bash
# train the model
cd train
python train_curapooling.py
```

## Citation
