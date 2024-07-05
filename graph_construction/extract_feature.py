import torch
import sys
import os
import openslide
from tqdm import tqdm
from file_utils import save_hdf5
from utils import collate_features
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_h5 import Whole_Slide_Bag_FP
from ctran import ctranspath


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'ctranspath.pth')
model.load_state_dict(td['model'], strict=True)
model.to(device)
model.eval()
if __name__ == "__main__":
    label = ''
    wsi_path = ''
    coords_path = '' #使用更新之后的坐标
    feats_dir = ''
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)
    patch_level=1
    patch_size=256

    error=0
    skip_num=0


    for slide in tqdm(os.listdir(coords_path)):

        id=slide.split('.')[0]
        if label=='0' or label=='1':
            slide_path = os.path.join(wsi_path, id + '.ibl.tiff')
            coords_name=id+'.ibl.h5'

        else:
            coords_name=id+'.h5'
            slide_path = os.path.join(wsi_path,id+'.svs')
        output_path = os.path.join(feats_dir,id+'.h5')
        if os.path.exists(output_path):
            print('exist')
            continue

        coords_file=os.path.join(coords_path,coords_name)

        # 找到slide的label
        # file_name, _ = os.path.splitext(os.path.splitext(slide)[0])
        # label_dicts = csv.DictReader(open(labels_file, 'r'))
        # bag_label = 0
        # for dict in label_dicts:
        #     if dict['file_name'] == file_name:
        #         bag_label = torch.tensor(int(dict['STAS']))
        #         # print('bag_label:',bag_label)

        wsi= openslide.OpenSlide(slide_path)
        dataset=Whole_Slide_Bag_FP(file_path=coords_file,patch_size=patch_size,patch_level=patch_level,wsi=wsi,pretrained=True,custom_downsample=1,custom_transforms=trnsfrms_val)
        kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset=dataset, batch_size=512, **kwargs, collate_fn=collate_features)
        cnt=0
        mode='w'
        for idx, (batch, coords) in enumerate(loader):
            batch=batch.to(device)
            with torch.no_grad():
                feats=model(batch)
            # if idx < 10:
            #     print(feats)
            feats=feats.cpu().numpy()
            asset_dict = {'features': feats, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
