import os
from pathlib import Path
import subprocess
from colmap.scripts.python.read_write_model import read_images_binary
from colmap.scripts.python.read_write_model import Image
import numpy as np
import json

import logging
logging.getLogger().setLevel(logging.INFO)

# https://colmap.github.io/format.html#images-txt
def get_coordinates_from_image(img:Image) -> np.array:
    Rt = img.qvec2rotmat().transpose() 
    T = img.tvec

    coordinates = np.matmul(-Rt, T)
    return coordinates

def fix_rotaion_in_space(array):
    x = -array[:,2]
    y = array[:,0]
    z = -array[:,1]
    return np.concatenate([[x],[y],[z]], axis=0).T


def main(location_id="311/918", scene_id="0"):
    image_bin_path = Path(os.path.join("./reconstruct", location_id, "colmap", scene_id, "images.bin"))
    
    if not image_bin_path.exists():
        return_code = subprocess.run(f"s5cmd --no-sign-request cp s3://megascenes/{str(image_bin_path)} ./{str(image_bin_path)}".split(" "), shell=True)
    else:
        logging.info(f"not downloading {image_bin_path} again.")
    
    all_bin_images = read_images_binary(image_bin_path)

    image_coordinates = []
    image_index = []
    for index, image in all_bin_images.items():
        image_coordinates.append(get_coordinates_from_image(image))
        image_index.append(index)

    image_dict = {idx: coords.tolist() for idx, coords in zip(image_index, image_coordinates)}
    image_coordinates = np.vstack(image_coordinates)
    logging.info(f"transformed {image_coordinates.shape} images")

    save_path = Path(f"./camera-location/{location_id}/{scene_id}/coordinates.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as outfile: 
        json.dump(image_dict, outfile, indent=4)
    
    logging.info(f"Done with loading location {location_id} and scene {scene_id}")

if __name__ == "__main__":
    main(location_id="311/918", scene_id="0")




