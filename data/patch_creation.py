import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import openslide
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm

TRAINING_ROOT = "/path/to/training/data"
PATCH_SIZE = 256
TISSUE_THRESHOLD = 0.7
TUMOR_OVERLAP_THRESHOLD = 0.5  # Minimum overlap to be labeled as tumor

NORMAL_DIR = os.path.join(TRAINING_ROOT, "normal")
OUTPUT_NORMAL = os.path.join(TRAINING_ROOT, "normal_patches")

TUMOR_DIR = os.path.join(TRAINING_ROOT, "tumor")
LESION_ANNOTATIONS_DIR = os.path.join(TRAINING_ROOT, "lesion_annotations")
OUTPUT_TUMOR = os.path.join(TRAINING_ROOT, "tumor_patches")
LABEL_FILE = os.path.join(TRAINING_ROOT, "train_tumor.txt")


def parse_xml_to_polygons(xml_path):
    """Parses an XML annotation file and returns lists of polygons for tumors and holes."""
    tumor_polys = []
    hole_polys = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for annotation in root.findall(".//Annotation"):
            group = annotation.get("PartOfGroup")
            coords = []
            for coord in annotation.findall(".//Coordinate"):
                x = float(coord.get("X"))
                y = float(coord.get("Y"))
                coords.append((x, y))
            
            if len(coords) > 2:
                poly = Polygon(coords)
                if group in ["_0", "_1"]:
                    tumor_polys.append(poly)
                elif group == "_2":
                    hole_polys.append(poly)
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error parsing {xml_path}: {e}")
    return tumor_polys, hole_polys


def create_normal_patches(slides_dir, output_dir, patch_size, tissue_threshold):
    """Generates patches from normal slides without annotations."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slide_files = [f for f in os.listdir(slides_dir) if not f.endswith(".xml")]

    for slide_file in tqdm(slide_files, desc="Processing normal slides"):
        slide_path = os.path.join(slides_dir, slide_file)
        try:
            slide = openslide.OpenSlide(slide_path)
        except openslide.OpenSlideError:
            print(f"Could not open {slide_file}, skipping.")
            continue

        lowest_res_level = slide.level_count - 1
        low_res_dims = slide.level_dimensions[lowest_res_level]
        low_res_img = slide.read_region((0, 0), lowest_res_level, low_res_dims).convert("L")
        
        low_res_np = np.array(low_res_img)
        _, binary_mask = cv2.threshold(low_res_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        downsample = slide.level_downsamples[lowest_res_level]
        slide_width, slide_height = slide.level_dimensions[0]
        
        for y in range(0, slide_height, patch_size):
            for x in range(0, slide_width, patch_size):
                mask_x, mask_y = int(x / downsample), int(y / downsample)
                mask_w, mask_h = int(patch_size / downsample), int(patch_size / downsample)
                if mask_x + mask_w > binary_mask.shape[1] or mask_y + mask_h > binary_mask.shape[0]:
                    continue
                
                mask_patch = binary_mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]

                if np.mean(mask_patch > 0) >= tissue_threshold:
                    if x + patch_size > slide_width or y + patch_size > slide_height:
                        continue
                        
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                    
                    patch_name = f"{os.path.splitext(slide_file)[0]}_x{x}_y{y}.png"
                    patch.save(os.path.join(output_dir, patch_name))
        
        slide.close()


def create_tumor_patches(slides_dir, annotations_dir, output_dir, label_file_path, patch_size, tissue_threshold, tumor_overlap_threshold):
    """Generates labeled patches from tumor slides with XML annotations."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slide_files = [f for f in os.listdir(slides_dir) if not f.endswith(".xml")]

    with open(label_file_path, "w") as label_file:
        for slide_file in tqdm(slide_files, desc="Processing tumor slides"):
            slide_path = os.path.join(slides_dir, slide_file)
            xml_path = os.path.join(annotations_dir, os.path.splitext(slide_file)[0] + ".xml")

            if not os.path.exists(xml_path):
                print(f"Annotation file not found for {slide_file} in {annotations_dir}, skipping.")
                continue

            try:
                slide = openslide.OpenSlide(slide_path)
            except openslide.OpenSlideError:
                print(f"Could not open {slide_file}, skipping.")
                continue
            
            tumor_polys, hole_polys = parse_xml_to_polygons(xml_path)
            if not tumor_polys:
                print(f"No tumor annotations found in {xml_path}, skipping slide.")
                slide.close()
                continue
            
            tumor_geom = unary_union(tumor_polys)
            if hole_polys:
                holes_geom = unary_union(hole_polys)
                tumor_geom = tumor_geom.difference(holes_geom)

            lowest_res_level = slide.level_count - 1
            downsample = slide.level_downsamples[lowest_res_level]
            low_res_dims = slide.level_dimensions[lowest_res_level]
            low_res_img = slide.read_region((0, 0), lowest_res_level, low_res_dims).convert("L")
            low_res_np = np.array(low_res_img)
            _, binary_mask = cv2.threshold(low_res_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            slide_width, slide_height = slide.level_dimensions[0]
            
            for y in range(0, slide_height, patch_size):
                for x in range(0, slide_width, patch_size):
                    mask_x, mask_y = int(x / downsample), int(y / downsample)
                    mask_w, mask_h = int(patch_size / downsample), int(patch_size / downsample)
                    if mask_x + mask_w > binary_mask.shape[1] or mask_y + mask_h > binary_mask.shape[0]:
                        continue
                    mask_patch = binary_mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]
                    if np.mean(mask_patch > 0) < tissue_threshold:
                        continue
                    
                    patch_box = box(x, y, x + patch_size, y + patch_size)
                    intersection_area = patch_box.intersection(tumor_geom).area
                    tumor_ratio = intersection_area / (patch_size * patch_size)
                    
                    label = 1 if tumor_ratio >= tumor_overlap_threshold else 0
                    
                    if x + patch_size > slide_width or y + patch_size > slide_height:
                        continue
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                    
                    patch_name = f"{os.path.splitext(slide_file)[0]}_x{x}_y{y}.png"
                    patch_path = os.path.join(output_dir, patch_name)
                    patch.save(patch_path)
                    
                    relative_path = os.path.join(os.path.basename(output_dir), patch_name)
                    label_file.write(f"{relative_path} {label}\n")
            
            slide.close()


if __name__ == "__main__":
    print(f"TRAINING_ROOT is set to: {TRAINING_ROOT}")

    if os.path.exists(NORMAL_DIR):
        print("Starting normal patch creation...")
        create_normal_patches(NORMAL_DIR, OUTPUT_NORMAL, PATCH_SIZE, TISSUE_THRESHOLD)
        print(f"Normal patch creation complete. Patches are in '{OUTPUT_NORMAL}'.")
    else:
        print(f"Normal slides directory not found at '{NORMAL_DIR}', skipping.")

    if os.path.exists(TUMOR_DIR):
        print("Starting tumor patch creation...")
        create_tumor_patches(TUMOR_DIR, LESION_ANNOTATIONS_DIR, OUTPUT_TUMOR, LABEL_FILE, PATCH_SIZE, TISSUE_THRESHOLD, TUMOR_OVERLAP_THRESHOLD)
        print(f"Tumor patch creation complete. Patches are in '{OUTPUT_TUMOR}'.")
        print(f"Label file is at '{LABEL_FILE}'.")
    else:
        print(f"Tumor slides directory not found at '{TUMOR_DIR}', skipping.")

    print("\nAll processing finished.")
