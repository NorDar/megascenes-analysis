import os
from pathlib import Path
import subprocess
from colmap.scripts.python.read_write_model import read_images_binary
from colmap.scripts.python.read_write_model import Image
import numpy as np
import json
import pickle

import logging
logging.getLogger().setLevel(logging.INFO)

# https://colmap.github.io/format.html#images-txt
def get_coordinates_from_image(img:Image) -> np.array:
    Rt = img.qvec2rotmat().transpose() 
    T = img.tvec

    coordinates = np.matmul(-Rt, T)
    return coordinates

def fix_rotaion_in_space(array, scene_id):
    if scene_id == "311/918": # Grossmünster
        x = -array[2]
        y = array[0]
        z = -array[1]
    else:
        x = -array[2]
        y = array[0]
        z = -array[1]
    
    return np.concatenate([[x],[y],[z]], axis=0).T



def download_reconstruction_data(location_id="311/918", reconstruction_id="0"):
    image_base_path = os.path.join("./reconstruct", location_id, "colmap", reconstruction_id)
    image_bin_path = Path(os.path.join(image_base_path, "images.bin"))
    
    if not image_bin_path.exists():
        return_code = subprocess.run(f"s5cmd --no-sign-request cp s3://megascenes/{str(image_bin_path)} {str(image_base_path)}/".split(" "), shell=False)
    else:
        logging.info(f"not downloading {image_bin_path} again.")
    
    all_bin_images = read_images_binary(image_bin_path)

    image_coordinates = []
    image_index = []
    image_path = []
    for index, image in all_bin_images.items():
        rotated_image = fix_rotaion_in_space(array=get_coordinates_from_image(image), scene_id=location_id)
        image_coordinates.append(rotated_image)
        image_index.append(index)
        image_path.append(image.name)

    
    logging.info(f"Rotated x, y, z according to sample transformation 311/918")

    image_dict = {idx: coords.tolist() for idx, coords in zip(image_index, image_coordinates)}
    image_coordinates = np.vstack(image_coordinates)
    logging.info(f"transformed {image_coordinates.shape} images")

    save_path = Path(f"./camera-location/{location_id}/{reconstruction_id}/coordinates.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as outfile: 
        json.dump(image_dict, outfile, indent=4)

    path_dict = {idx: path for idx, path in zip(image_index, image_path)}
    save_path_image_paths = Path(f"./image-paths/{location_id}/{reconstruction_id}/paths.json")
    save_path_image_paths.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path_image_paths, "w", encoding='utf8') as outfile: 
        json.dump(path_dict, outfile, indent=4, ensure_ascii=False)
    
    logging.info(f"Done with loading location {location_id} and reconstruction {reconstruction_id}")

def download_images(location_id="311/918", reconstruction_id="0", outlier_only=True):
    logging.info(f"Staring to download images for {location_id}, reconstruction {reconstruction_id}")
    image_paths = Path(f"./image-paths/{location_id}/{reconstruction_id}/paths.json")
    with open(image_paths, "r") as file:
        paths = json.load(file)
    
    logging.info(f"Found {len(paths)} images")

    if outlier_only:
        len_total = len(paths)
        with open(f"./image-paths/{location_id}/{reconstruction_id}/outlier_list.pickle", "rb") as file:
            outliers = pickle.load(file)
        paths = {key: value for key, value in paths.items() if int(key) in outliers}
        logging.info(f"Only loading {len(paths)} outliers from total of {len_total} images.")

    for idx, image_path in paths.items():
        single_path = Path(f"images/{location_id}/{image_path}")
        if not single_path.exists():
            #return_code = subprocess.run(f"s5cmd --no-sign-request cp s3://megascenes/{str(single_path)} ./{str(single_path)}".split(" "), shell=True)
            os.system(f's5cmd --no-sign-request cp "s3://megascenes/{str(single_path)}" "./{str(single_path)}"')
        else:
            logging.info(f"Not loading existing {single_path}")

    logging.info(f"Finished downloading {len(paths)} images.")

if __name__ == "__main__":
    download_reconstruction_data(location_id="311/918", reconstruction_id="0")
    download_images(location_id="311/918", reconstruction_id="0", outlier_only=True)




