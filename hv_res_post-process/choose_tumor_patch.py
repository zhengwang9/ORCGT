import json
import os
from tqdm import tqdm
import glob

def load_json_info(json_path):

    bbox_list = []
    centroid_list = []
    contour_list = [] 
    type_list = []

    try:
        with open(json_path) as json_file:
            data = json.load(json_file)
            mag_info = data['mag']
            nuc_info = data['nuc']
            for inst in nuc_info:
                inst_info = nuc_info[inst]
                inst_centroid = inst_info['centroid']
                centroid_list.append(inst_centroid)
                inst_contour = inst_info['contour']
                contour_list.append(inst_contour)
                inst_bbox = inst_info['bbox']
                bbox_list.append(inst_bbox)
                inst_type = inst_info['type']
                type_list.append(inst_type)
    except Exception as e:
        print(json_path)
    return centroid_list,contour_list, bbox_list, type_list


def get_tumor_patch_index(people_dir,tumor_patch_index):
    tumor_threshold = 5
    json_path = people_dir + '/' + 'json'
    json_files = os.listdir(json_path)

    for json_file in tqdm(json_files):
        patient_name = json_file[:json_file.rindex('_')]
        if patient_name not in tumor_patch_index.keys():
            tumor_patch_index[patient_name] = []
        patch_index = int(json_file[json_file.rindex('p')+1:json_file.rindex('.')])
        _,_,_,type_list = load_json_info(os.path.join(json_path,json_file))
        tumor_num = type_list.count(1)

        if tumor_num >= tumor_threshold:
            tumor_patch_index[patient_name].append(patch_index)
    return tumor_patch_index
    
    

if __name__ == "__main__":
    people_root = '../hovernet_result'
    people_dir_list = glob.glob(people_root + '/*')
    for people_dir in people_dir_list:
        tumor_patch_index = {}
        print(people_dir)
        id = people_dir[people_dir.rindex('/')+1:]
        if os.path.exists('./result/{}.json'.format(id)):
            print(people_dir, 'done!!!!!!!')
        tumor_patch_index = get_tumor_patch_index(people_dir,tumor_patch_index)
        json_file_path = './result/{}.json'.format(id)
        with open(json_file_path, 'w') as json_file:
            json.dump(tumor_patch_index, json_file, indent=4)
        print(f"Dictionary saved to {json_file_path}")




