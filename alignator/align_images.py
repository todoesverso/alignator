#!/usr/bin/env python

import argparse
import concurrent.futures
import logging
import os
import time

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import (
    structural_similarity,
    mean_squared_error,
    normalized_root_mse,
)
from tqdm import tqdm
import subprocess
import shlex
from collections import defaultdict

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

max_features = 10000
feature_retention = 0.15
diff_threshold = 5000
threaded = False
analize = False
images_discarded = defaultdict()
images_discarded_total = defaultdict()
scores = []


def run_command(cmd, log_output=True):
    cmd = shlex.split(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if log_output:
        for line in p.stdout:
            logger.info(line.decode("utf8").strip())
    logger.debug("Command finished with status %d", p.poll())


def getArgs():
    parser = argparse.ArgumentParser(description="Coso")
    parser.add_argument(
        "-t", "--threaded", help="run in threads", default=False, action="store_true"
    )
    parser.add_argument(
        "-a",
        "--analize",
        help="only analize images",
        default=False,
        action="store_true",
    )
    parser.add_argument("-p", "--path", help="path with images", required=True)
    parser.add_argument("-d", "--dest", help="destinatin path", required=True)
    parser.add_argument(
        "-v", "--video_dest", help="video destination path", required=True
    )
    parser.add_argument("-bi", "--baseimage", help="image to match", required=True)
    parser.add_argument(
        "-mf",
        "--max_features",
        help="maximum number of features to consider",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "-fr",
        "--feature_retention",
        help="fraction of features to retain",
        default=0.15,
    )
    parser.add_argument(
        "-dt",
        "--diff_threshold",
        help="discard different images",
        default=5000,
    )
    parser.add_argument(
        "-r",
        "--framerate",
        help="video framerate",
        default=4,
    )

    return parser.parse_args()


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = _convert_to_grayscale(image)

    # Calculate grayscale histogram
    hist = _get_hist(gray)
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def histoImprove(image, gridsize=10, clip_limit=1.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def featureAlign(image_path, image, base_image, kp, desc):
    # Convert images to grayscale
    image_gray = _convert_to_grayscale(image)

    # Detect ORB features and compute descriptors.
    keypoints1, descriptors1 = _detector(image_path, image_gray, mf=int(max_features))
    keypoints2, descriptors2 = kp, desc

    # Match features.
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # matches = matcher.match(descriptors1, descriptors2, None)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * float(feature_retention))
    matches = matches[:numGoodMatches]
    logger.debug("Image=%s, Features good matches %d", image_path, len(matches))

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    # h, mask = cv2.findHomography(points1, points2, cv2.LEAST_MEDIAN, 5.0)

    # Use homography
    height, width, channels = base_image.shape
    image_aligned = cv2.warpPerspective(image, h, (width, height))

    return image_aligned, len(matches)


def get_image_kp_and_desc(image_path, mf=5000):
    image = _read_image(image_path)
    gray_image = _convert_to_grayscale(image)
    kp, desc = _detector(image_path, gray_image, mf=mf)
    logger.info(
        "Reading image=%s keypoints=%s descriptors=%s maxfeatures=%s",
        image_path,
        len(kp),
        len(desc),
        mf,
    )

    return image, kp, desc


def get_images_path(_path):
    if not os.path.isdir(_path):
        logger.exception("path=%s is not a dir", _path)
    imgs = []
    for file in os.listdir(_path):
        ext = file.split(".")[-1]
        if ext in ("jpeg, jpg, png"):
            imgs.append(os.path.join(_path, file))
    return imgs


def alignImages(base_image_path, source_images_path, dest_path):

    base_image, kp, desc = get_image_kp_and_desc(base_image_path, max_features)
    base_image_gray = _convert_to_grayscale(base_image)
    images = get_images_path(source_images_path)
    discarded = 0

    if threaded:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for image_path in images:
                futures.append(
                    executor.submit(
                        process,
                        base_image,
                        base_image_gray,
                        kp,
                        desc,
                        image_path,
                        dest_path,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                discarded += future.result()
    else:
        for image_path in images:
            discarded += process(base_image, base_image_gray, kp, desc, image_path, dest_path)

    return discarded


def apply_vignette(img):
    rows, cols = img.shape[:2]
    zeros = np.copy(img)
    zeros[:, :, :] = 0
    sigma = 1200
    a = cv2.getGaussianKernel(cols, sigma)
    b = cv2.getGaussianKernel(rows, sigma)
    c = b * a.T
    d = c / c.max()
    zeros[:, :, 0] = img[:, :, 0] * d
    zeros[:, :, 1] = img[:, :, 1] * d
    zeros[:, :, 2] = img[:, :, 2] * d

    return zeros


def process(base_image, base_image_gray, kp, desc, image_path, dest_path):
    logger.debug("Processing %s", image_path)
    image = _read_image(image_path)
    original_image = _read_image(image_path) if not analize else None

    image, _ = featureAlign(image_path, image, base_image, kp, desc)
    dest_image_gray = _convert_to_grayscale(image)
    score = mean_squared_error(
        base_image_gray,
        dest_image_gray,
    )

    if score > diff_threshold:
        # If the image is not alignable lets just use the original one and hope for the best
        image = original_image
        logger.debug("Image discarded (score %d): %s", score, image_path)
        images_discarded[image_path] = score
        scores.append(score)

    if analize:
        if score > diff_threshold:
            return 1
        else:
            return 0

    image = automatic_brightness_and_contrast(image)
    image = histoImprove(image)
    image = apply_vignette(image)
    image_dest = _build_dest_path(dest_path, image_path, score)
    logger.debug(image_dest)
    cv2.imwrite(image_dest, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if score > diff_threshold:
        return 1
    else:
        return 0




def create_video(img_path, framerate=4, dest="out.mp4"):
    cmd = f"ffmpeg -r {framerate} -pattern_type glob -i '{img_path}/*.jpg' -c:v  libx264 -pix_fmt yuvj422p {dest}"
    run_command(cmd)


def _build_dest_path(dest_path, image_path, score):
    image_dest = os.path.join(
        dest_path,
        os.path.basename(os.path.splitext(image_path)[0])
        + "_"
        + str(int(score))
        + ".jpg",
    )
    logger.debug("Dest: %s", image_dest)
    return image_dest


def _read_image(image_path):
    return cv2.imread(image_path)


def _convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _detector(image_path, img, mf=5000):
    detect = cv2.ORB_create(mf)
    # detect = cv2.SIFT_create(mf) # Not so good
    kp, des = detect.detectAndCompute(img, None)
    logger.info(
        "detector: image=%s keypoints=%s descriptors=%s maxfeatures=%s", image_path, len(kp), len(des), mf
    )
    return (kp, des)


def _get_hist(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])


if __name__ == "__main__":

    args = getArgs()

    max_features = args.max_features
    feature_retention = args.feature_retention
    diff_threshold = args.diff_threshold
    threaded = args.threaded
    analize = args.analize
    video_dest = args.video_dest
    framerate = args.framerate

    if analize:
        images = get_images_path(args.path)
        x = 1
        images_count = len(images)
        for i in images:
            start = time.time()
            logger.info("-> Image (%s) %d/%d", i, x, images_count)
            images_discarded_total[i] = alignImages(i, args.path, args.dest)
            x += 1
            end = time.time()
            elapsed = end - start
            logger.info(f" --> Execution took: {elapsed} s")
            logger.info("  --> Partial Images discarded: ")
            logger.info(images_discarded_total)
    else:
        alignImages(args.baseimage, args.path, args.dest)

    if not analize:
        create_video(args.dest, framerate=framerate, dest=video_dest)

    scores_np = np.array(scores)  # , dtype=np.uint32)
    df_describe = pd.DataFrame(scores_np)
    logger.info(df_describe.describe())
    logger.info(scores)

    logger.info("Images discarded: ")
    logger.info(images_discarded_total)
