# %%
import json
import os
from tqdm import tqdm
import h5py

label = 0
json_file_path = ''.format(label)
# 特征路径 坐标路径
fea_path_root = ''.format(label)
cood_root = ''.format(label)
fea_new_root = ''.format(label)
if not os.path.exists(fea_new_root):
    os.makedirs(fea_new_root)

with open(json_file_path) as json_file:
    tumor_patch_index = json.load(json_file)

# %%
# 更新坐标和feature
coord_pathls = os.listdir(cood_root)

for patient_name in tqdm(coord_pathls):

    patient_name = patient_name[:patient_name.index('.')]
    if patient_name not in tumor_patch_index.keys():
        continue
    coord_path =  os.path.join(cood_root, patient_name + ".ibl.h5")
    if os.path.exists(coord_path):
        coord_path =  os.path.join(cood_root, patient_name + ".h5") #svs
    fea_path = os.path.join(fea_path_root, patient_name + ".ibl.tiff.h5")
    if not os.path.exists(fea_path):
        fea_path = os.path.join(fea_path_root, patient_name + ".h5") #svs

    patches=h5py.File(fea_path,'r')
    coords = patches['coords'][:]
    features = patches['features'][:]

    tumor_index = tumor_patch_index[patient_name]
    tumor_coords = [coords[i] for i in tumor_index]
    tumor_fea = [features[i] for i in tumor_index]
    
    patches_tumor = {}
    patches_tumor['coords'] = tumor_coords
    patches_tumor['features'] = tumor_fea
    # 指定保存的 HDF5 文件路径
    # hdf5_file_path = os.path.join(fea_new_root, patient_name + ".ibl.h5")
    hdf5_file_path = os.path.join(fea_new_root, patient_name + ".ibl.h5")
    if os.path.exists(hdf5_file_path):
        continue

    # 使用 h5py 创建文件并写入数据
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        # 遍历字典的键值对，并将数据保存到 HDF5 文件中
        for key, value in patches_tumor.items():
            if isinstance(value, list):
                # 如果值是列表，则将其保存为 HDF5 数据集
                hdf5_file.create_dataset(key, data=value)
            else:
                # 否则，将其保存为 HDF5 属性
                hdf5_file.attrs[key] = value

    print(f"Data saved to {hdf5_file_path}")

