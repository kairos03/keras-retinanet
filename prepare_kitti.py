#!/usr/bin/env python2
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
"""
Prepares KITTI data for ingestion by DIGITS
"""

import argparse
import os
import re
import shutil
import zipfile

import cv2
import numpy

def extract_data(input_dir, output_dir):
    """
    Extract zipfiles at input_dir into output_dir
    """
    if os.path.isdir(output_dir):
        print ('  Using extracted data at %s.' % output_dir)
        return

    for filename in (
            'data_object_label_2.zip',
            'data_object_image_2.zip',
            'data_object_image_3.zip',
            'devkit_object.zip'):
        filename = os.path.join(input_dir, filename)
        zf = zipfile.ZipFile(filename, 'r')
        print('Unzipping %s ...' % filename)
        zf.extractall(output_dir)


def get_image_to_video_mapping(devkit_dir):
    """
    Return a mapping from image filename (e.g. 7282 which is training/image_2/007282.png)
        to video and frame (e.g. {'video': '2011_09_26_0005', 'frame': 109})
    """
    image_to_video = {}
    mapping_lines = None
    with open(os.path.join(devkit_dir, 'mapping', 'train_mapping.txt'), 'r') as infile:
        mapping_lines = infile.readlines()

    with open(os.path.join(devkit_dir, 'mapping', 'train_rand.txt'), 'r') as infile:
        for image_index, mapping_index in enumerate(infile.read().split(',')):
            mapping_index = mapping_index.strip()
            if not mapping_index:
                continue
            mapping_index = int(mapping_index) - 1
            map_line = mapping_lines[mapping_index]
            match = re.match('^\s*[\d_]+\s+(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync\s+(\d+)$\s*$', map_line)
            if not match:
                raise ValueError('Unrecognized mapping line "%s"' % map_line)
            date = match.group(1)
            video_id = match.group(2)
            video_name = '%s_%s' % (date, video_id)
            frame_index = int(match.group(3))
            if image_index in image_to_video:
                raise ValueError('Conflicting mappings for image %s' % image_index)
            image_to_video[image_index] = {
                'video': video_name,
                'frame': frame_index,
            }

    return image_to_video


def stereo_to_disparity(left_path, right_path, stereo_bm=None):
    if stereo_bm is None:
        stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    
    imgL = cv2.imread(left_path, 0)
    imgR = cv2.imread(right_path, 0)

    disparity = stereo_bm.compute(imgL, imgR)

    return disparity

def make_disparity(base_dir, replace=False):
    disparity_dir = os.path.join(base_dir, 'disparity')
    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)
    
    stereo_bm = cv2.StereoSGBM_create(minDisparity=-64, numDisparities=256, blockSize=5, P1=600, P2=2400, disp12MaxDiff=10, preFilterCap=4, uniquenessRatio=1, speckleWindowSize=150, speckleRange=2)
    
    for image_fname in os.listdir(os.path.join(base_dir, 'image_2')):
        left_path = os.path.join(base_dir, 'image_2', image_fname)
        right_path = os.path.join(base_dir, 'image_3', image_fname)

        disp = stereo_to_disparity(left_path, right_path, stereo_bm=stereo_bm)        
        cv2.imwrite(os.path.join(disparity_dir, image_fname), disp)

def split_by_video(training_dir, mapping, split_dir,
                   use_symlinks=True, disparity=True):
    """
    Create one directory per video in split_dir
    """
    new_images_dir = os.path.join(split_dir, 'images')
    new_labels_dir = os.path.join(split_dir, 'labels')
    if disparity:
        new_disp_dir = os.path.join(split_dir, 'disparity')

    if os.path.isdir(new_images_dir):
        shutil.rmtree(new_images_dir)
    if os.path.isdir(new_labels_dir):
        shutil.rmtree(new_labels_dir)
    if disparity:
        if os.path.isdir(new_disp_dir):
            shutil.rmtree(new_labels_dir)

    for old_image_fname in os.listdir(os.path.join(training_dir, 'image_2')):
        old_image_path = os.path.abspath(os.path.join(training_dir, 'image_2', old_image_fname))
        image_index_str, image_ext = os.path.splitext(
            os.path.basename(old_image_fname))
        image_index_int = int(image_index_str)
        video_name = mapping[image_index_int]['video']
        frame_id = '%09d' % mapping[image_index_int]['frame']

        # Copy image
        new_image_dir = os.path.join(new_images_dir, video_name)
        if not os.path.isdir(new_image_dir):
            os.makedirs(new_image_dir)
        new_image_fname = '%s_%s%s' % (frame_id, image_index_str, image_ext)
        new_image_path = os.path.join(new_image_dir, new_image_fname)
        if use_symlinks:
            os.symlink(old_image_path, new_image_path)
        else:
            shutil.copyfile(old_image_path, new_image_path)

        # Copy right image
        if disparity:
            old_image_path = os.path.abspath(os.path.join(training_dir, 'disparity', old_image_fname))
            image_index_str, image_ext = os.path.splitext(
                os.path.basename(old_image_fname))
            image_index_int = int(image_index_str)
            video_name = mapping[image_index_int]['video']
            frame_id = '%09d' % mapping[image_index_int]['frame']

            # Copy image
            new_image_dir = os.path.join(new_disp_dir, video_name)
            if not os.path.isdir(new_image_dir):
                os.makedirs(new_image_dir)
            new_image_fname = '%s_%s%s' % (frame_id, image_index_str, image_ext)
            new_image_path = os.path.join(new_image_dir, new_image_fname)
            if use_symlinks:
                os.symlink(old_image_path, new_image_path)
            else:
                shutil.copyfile(old_image_path, new_image_path)

        # Copy label
        old_label_fname = '%s.txt' % image_index_str
        old_label_path = os.path.abspath(os.path.join(training_dir, 'label_2', old_label_fname))
        new_label_fname = '%s_%s.txt' % (frame_id, image_index_str)
        new_label_dir = os.path.join(new_labels_dir, video_name)
        if not os.path.isdir(new_label_dir):
            os.makedirs(new_label_dir)
        new_label_path = os.path.join(new_label_dir, new_label_fname)
        if use_symlinks:
            os.symlink(old_label_path, new_label_path)
        else:
            shutil.copyfile(old_label_path, new_label_path)


def split_for_training(split_dir, train_dir, val_dir,
                       use_symlinks=True, disparity=True):
    """
    Create directories of images for training and validation
    """
    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    if os.path.isdir(val_dir):
        shutil.rmtree(val_dir)

    for images_dirname in os.listdir(os.path.join(split_dir, 'images')):
        match = re.match('^(\d{4})_(\d{2})_(\d{2})_(\d+)$', images_dirname)
        if not match:
            raise ValueError('Unrecognized format of directory named "%s"' % images_dirname)
        # year = int(match.group(1))
        month = int(match.group(2))
        date = int(match.group(3))
        video_id = int(match.group(4))

        # Filter out some videos for the validation set
        # XXX this is pretty arbitrary
        if month == 9 and date == 26 and video_id <= 18:
            output_dir = val_dir
        else:
            output_dir = train_dir

        # Copy images
        old_images_dir = os.path.join(split_dir, 'images', images_dirname)
        new_images_dir = os.path.join(output_dir, 'images')
        if not os.path.isdir(new_images_dir):
            os.makedirs(new_images_dir)
        for fname in os.listdir(old_images_dir):
            old_image_path = os.path.realpath(os.path.join(old_images_dir, fname))
            new_image_path = os.path.join(new_images_dir, os.path.basename(old_image_path))
            if use_symlinks:
                os.symlink(old_image_path, new_image_path)
            else:
                shutil.move(old_image_path, new_image_path)

        # Copy labels
        old_labels_dir = os.path.join(split_dir, 'labels', images_dirname)
        new_labels_dir = os.path.join(output_dir, 'labels')
        if not os.path.isdir(new_labels_dir):
            os.makedirs(new_labels_dir)
        for fname in os.listdir(old_labels_dir):
            old_label_path = os.path.realpath(os.path.join(old_labels_dir, fname))
            new_label_path = os.path.join(new_labels_dir, os.path.basename(old_label_path))
            if use_symlinks:
                os.symlink(old_label_path, new_label_path)
            else:
                shutil.move(old_label_path, new_label_path)

        # right image
        # Copy images
        if disparity:
            old_images_dir = os.path.join(split_dir, 'disparity', images_dirname)
            new_images_dir = os.path.join(output_dir, 'disparity')
            if not os.path.isdir(new_images_dir):
                os.makedirs(new_images_dir)
            for fname in os.listdir(old_images_dir):
                old_image_path = os.path.realpath(os.path.join(old_images_dir, fname))
                new_image_path = os.path.join(new_images_dir, os.path.basename(old_image_path))
                if use_symlinks:
                    os.symlink(old_image_path, new_image_path)
                else:
                    shutil.move(old_image_path, new_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare KITTI data')
    parser.add_argument('-i', '--input-dir', default='',
                        help='The directory containing the original KITTI zipfiles [default "./"]')
    parser.add_argument('-o', '--output-dir', default='kitti-data',
                        help='The output directory [default "./kitti-data"]')
    parser.add_argument('-s', '--no-symlinks', action='store_true',
                        help='Copy files instead of making symlinks')

    args = parser.parse_args()

    print('Extracting zipfiles ...')
    extract_data(

        args.input_dir,
        os.path.join(args.output_dir, 'raw'),
    )
    print('Calculating image to video mapping ...')
    mapping = get_image_to_video_mapping(
        os.path.join(args.output_dir, 'raw'),
    )
    print('Make disparity...')
    make_disparity(
        os.path.join(args.output_dir, 'raw', 'training'),
    )
    print('Splitting images by video ...')
    split_by_video(
        os.path.join(args.output_dir, 'raw', 'training'),
        mapping,
        os.path.join(args.output_dir, 'video-split'),
        use_symlinks=(not args.no_symlinks),
        disparity=True
    )
    print('Creating train/val split ...')
    split_for_training(
        os.path.join(args.output_dir, 'video-split'),
        os.path.join(args.output_dir, 'train'),
        os.path.join(args.output_dir, 'val'),
        use_symlinks=(not args.no_symlinks),
        disparity=True
    )
print('Done.')
