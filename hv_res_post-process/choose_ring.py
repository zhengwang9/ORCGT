import os
import h5py
import numpy
import cv2
import json
from shapely.geometry import Point, Polygon
from PIL import Image
import torch
import math


def point_to_line_distance(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if line_length == 0:
        return 0 

    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length ** 2)
    t = max(0, min(1, t))
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)

    distance = math.sqrt((x - projection_x) ** 2 + (y - projection_y) ** 2)
    return distance


def point_to_polygon_distance(point, polygon):
    distances = []
    for i in range(len(polygon)):
        current_point = polygon[i]
        next_point = polygon[(i + 1) % len(polygon)]
        distance = point_to_line_distance(point, current_point, next_point)
        distances.append(distance)
    return min(distances)


def compute_dist(polygon, coord):
    return point_to_polygon_distance(coord, polygon)



def choose_neighbor_out_in(threshold_out,threshold_in,slide_id,masks_path,coords_path):
    id = slide_id.split('.')[0]
    mask_path = os.path.join(masks_path, slide_id)
    mask = Image.open(mask_path)
    h5_id = id + '.ibl.h5'

    coords_path=''
    coords_file = os.path.join(coords_path, h5_id)
    coords = h5py.File(coords_file, 'r')
    coords = coords['coords'][:].reshape(-1,2)

    x_min, x_max = coords.T[0].min(), coords.T[0].max()
    y_min, y_max = coords.T[1].min(), coords.T[1].max()
    w, h = x_max - x_min, y_max - y_min
    h = int(h / 256) + 1
    w = int(w / 256) + 1

    x0 = (512 - w) // 2
    y0 = (512 - h) // 2
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    mask = numpy.array(mask).astype(numpy.uint8)
    mask = mask[y0:y0 + h, x0:x0 + w]

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    if areas == []:
        return False
    sorted_contours_areas = sorted(zip(contours, areas), key=lambda x: x[1], reverse=True)

    sorted_contours, sorted_areas = zip(*sorted_contours_areas)
    boundarys = []
    if sorted_areas[1] * 10 < sorted_areas[0]:
        k = 1
    else:
        k = 2
    for i in range(k):
        # print(k - i - 1)
        contour = sorted_contours[i]
        boundary_coords = []

        for point in contour:
            x, y = point[0]
            boundary_coords.append((x, y))
        boundarys.append(boundary_coords)


    patch_list = []
    for idx, (x, y) in enumerate(coords):
        x,y = int((x - x_min)/256), int((y - y_min)/256)
        tar = Point(x, y)
        dist_out = []
        dist_in = []
        for boundary in boundarys:
            polygon = Polygon(boundary)
            if polygon.contains(tar):
                # flag=1
                dist_in.append(compute_dist(boundary, (x, y)))
            else:
                dist_out.append(polygon.distance(tar))

        if dist_out != []:
            dist_out = min(dist_out)
        if dist_in != []:
            dist_in = min(dist_in)

        if (dist_out != [] and dist_out <= threshold_out) or (dist_in != [] and dist_in <= threshold_in):
            patch_list.append(idx)
    return [id,patch_list]




if __name__ == "__main__":
    masks_path = ''
    coords_path =''
    neighbor_patch = {}
    threshold_out = 40
    threshold_in = 15

    async_count = 16
    outputs = []

    list_dir = os.listdir(masks_path)
    for brunch in range(len(list_dir) // async_count+1):

        cstart = brunch * async_count

        ctx = torch.multiprocessing.get_context("spawn")
        pool = ctx.Pool(async_count)

        for slide in range(async_count):
            if (brunch * async_count + slide)<len(list_dir):
                slide_id =list_dir[brunch * async_count + slide]
                print(slide_id)
                output=pool.apply_async(choose_neighbor_out_in, args = (threshold_out,threshold_in,slide_id,masks_path,coords_path))
                outputs.append(output)
            else:
                break
        pool.close()
        pool.join()
    for output in outputs:
        result = output.get()
        if result!=False:
            neighbor_patch[result[0]] = result[1]
    json_file = json.dumps(neighbor_patch)
    save_file = f'./result/{threshold_out}_{threshold_in}.json'
    with open(save_file, 'w') as f:
        f.write(json_file)
