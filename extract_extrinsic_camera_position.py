import os
from pathlib import Path
import subprocess
from colmap.scripts.python.read_write_model import read_images_binary, read_cameras_binary
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
    cam_bin_path = Path(os.path.join(image_base_path, "cameras.bin"))
    
    if not image_bin_path.exists() or not cam_bin_path.exists():
        return_code = subprocess.run(f"s5cmd --no-sign-request cp s3://megascenes/{str(image_bin_path)} {str(image_base_path)}/".split(" "), shell=False)
        return_code = subprocess.run(f"s5cmd --no-sign-request cp s3://megascenes/{str(cam_bin_path)} {str(image_base_path)}/".split(" "), shell=False)
    else:
        logging.info(f"not downloading {image_bin_path} and {cam_bin_path} again.")
    
    logging.info(f"Reading binary data.")
    all_bin_images = read_images_binary(image_bin_path)
    all_bin_cameras = read_cameras_binary(cam_bin_path)

    image_dict, image_index, image_path = coordinates_per_image(all_bin_images, location_id)
    intrinsics_dict = intrinsics_per_image(all_bin_images, all_bin_cameras, location_id)

    # Save coordinates
    save_path = Path(f"./camera-location/{location_id}/{reconstruction_id}/coordinates.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as outfile: 
        json.dump(image_dict, outfile, indent=4)

    # Save intrinsics
    intrinsics_save_path = Path(f"./camera-location/{location_id}/{reconstruction_id}/intrinsics.json")
    intrinsics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(intrinsics_save_path, "w") as outfile: 
        json.dump(intrinsics_dict, outfile, indent=4)

    # Save image paths
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
            logging.debug(f"Not loading existing {single_path}")

    logging.info(f"Finished downloading {len(paths)} images.")


def radians_from_rotation_matrix(rotation_matrix):
    '''
    For a cameras rotation matrix we calculate angels based on 
    each separate rotation: Rx(psi), Ry(theta), Rz(phi)

    Returns radians
    '''
    # https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    R = rotation_matrix

    if np.isclose(R[2,1], 1) or np.isclose(R[2,1], -1):
        logging.debug("degenerate case of cos(theta) = 0")
        phi = 0 # can be anything...
        if np.islcose(R[2,0], -1):
            theta = np.pi/2
            psi = phi + np.atan2(R[0,1], R[0,2])
        else:
            theta = -np.pi/2
            psi = -phi + np.atan2(-R[0,1], -R[0,2])
    else:
        logging.debug("non-degenerate case where cos(theta) != 0")
        theta1 = -np.asin(R[2,0])
        theta2 = np.pi - theta1

        psi1 = np.atan2(R[2,1]/np.cos(theta1), R[2,2]/np.cos(theta1))
        psi2 = np.atan2(R[2,1]/np.cos(theta2), R[2,2]/np.cos(theta2))

        phi1 = np.atan2(R[1,0]/np.cos(theta1), R[0,0]/np.cos(theta1))
        phi2 = np.atan2(R[1,0]/np.cos(theta2), R[0,0]/np.cos(theta2))

        psi = psi1
        theta = theta1
        phi = phi1

    return psi, theta, phi

# https://stackoverflow.com/questions/77655360/direction-vector-correctly-scale-sinradiansx-cosradiansz-with-y-axis
def radians_to_unit_coord(radian_rotaion, location_id):
    ''' 
    Calculate the point on the unit sphere around the intrinsic origin
    based on the direction the camera is looking

    Includes a correction for different coordinate system like 
    extract_extrinsic_campera_position.fix_rotaion_in_space
    '''
    dir_x = np.sin(radian_rotaion[2]) * np.cos(radian_rotaion[1])
    dir_y = np.sin(radian_rotaion[1])
    dir_z = np.cos(radian_rotaion[2]) * np.cos(radian_rotaion[1])

    vector = (dir_x, dir_y, dir_z)
    uv =  vector / np.linalg.norm(vector)
    
    rotated_vector = fix_rotaion_in_space(array=uv, scene_id=location_id)
    return rotated_vector

def intrinsics_per_image(all_bin_images, all_bin_cameras, location_id):
    '''Camera params are f, cx, cz, k'''
    # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L281
    logging.info(f"Calculating the camera intrinsics")

    unit_vectors = []
    focal_lengths = []
    image_index = []
    camera_ids = []
    #for idx, cam in all_bin_cameras.items():
    for idx, image in all_bin_images.items():
        cam = all_bin_cameras[image.camera_id]
        assert cam.model == 'SIMPLE_RADIAL', f"Camera model is not 'SIMPLE_RADIAL'. Focal length might be at a different parameter."

        unit_vectors.append(radians_to_unit_coord(radians_from_rotation_matrix(image.qvec2rotmat()), location_id))
        focal_lengths.append(cam.params[0])
        image_index.append(idx)
        camera_ids.append(image.camera_id)

    camera_intrinsics = {image_index: 
        {
            "unit_vector_x": vector.tolist()[0],
            "unit_vector_y": vector.tolist()[1],
            "unit_vector_z": vector.tolist()[2],
            "focal_length": float(length), 
            "camera_id": cam_id
        }
        for image_index, vector, length, cam_id in zip(image_index, unit_vectors, focal_lengths, camera_ids)
    }

    return camera_intrinsics

def coordinates_per_image(all_bin_images, location_id):
    logging.info(f"Calculating the image coordinates")
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

    return image_dict, image_index, image_path

if __name__ == "__main__":
    download_reconstruction_data(location_id="311/918", reconstruction_id="0")
    download_images(location_id="311/918", reconstruction_id="0", outlier_only=True)




