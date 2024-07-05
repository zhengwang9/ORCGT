# %%
import json
import os
import h5py
from tqdm import tqdm
import glob

def get_neighbors_left_top(coord, step=512):
    x = coord[0]
    y = coord[1]
    return [
        (x - step, y - step), (x, y - step), (x + step, y - step),
        (x - step, y),                              (x + step, y),
        (x - step, y + step), (x, y + step), (x + step, y + step)
    ]

def tumor_around(coord, coords, step):
    coord_around = get_neighbors_left_top(coord, step)
    tumor_compute = 0
    for i in coord_around:
        if list(i) in coords:
            tumor_compute += 1
    
    return tumor_compute

# %%       
if __name__ == '__main__':

    json_root = './hv_res_post-process/result'
    json_file_list = glob.glob(json_root + '/*')
    tumor_patch_index = {}
    for json_file_path in tqdm(json_file_list):
        with open(json_file_path) as json_file:
            tumor_patch_index.update(json.load(json_file))

    #Features and coordinates root
    fea_path_root = ''
    cood_root = ''

    big_tumor_patch_index = {}
    small_tumor_patch_index = {}
    for patient_name in tqdm(tumor_patch_index.keys()):
        tumor_patches = tumor_patch_index[patient_name]
        coord_path =  os.path.join(cood_root, patient_name + ".h5")
        fea_path = os.path.join(fea_path_root, patient_name + ".h5")
        if patient_name not in big_tumor_patch_index:
            big_tumor_patch_index[patient_name] = []
        if patient_name not in small_tumor_patch_index:
            small_tumor_patch_index[patient_name] = []

        patches=h5py.File(fea_path,'r')
        coords = patches['coords'][:].tolist()
        coords_tumor = [coords[ind] for ind in tumor_patches]
        for idx in tumor_patches:
            coord = coords[idx]
            tumor_num = tumor_around(coord, coords_tumor, step=512)
            if tumor_num > 2:
                big_tumor_patch_index[patient_name].append(idx)
            else:
                small_tumor_patch_index[patient_name].append(idx)

    json_file_path = './result/tumor_patch_type_index_big.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(big_tumor_patch_index, json_file, indent=4)
    print(f"Dictionary saved to {json_file_path}")

    json_file_path = './result/tumor_patch_type_index_small.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(small_tumor_patch_index, json_file, indent=4)
    print(f"Dictionary saved to {json_file_path}")


