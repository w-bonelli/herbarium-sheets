import time
from math import ceil
from os.path import join

import click
import cv2
import imageio
import numpy as np
import skimage

import thresholding
from options import AnalysisOptions
from results import AnalysisResult
from utils import write_results


def process(options: AnalysisOptions) -> AnalysisResult:
    output_prefix = join(options.output_directory, options.input_stem)
    print(f"Extracting traits from '{options.input_name}'")

    # read grayscale image
    gray_image = imageio.imread(options.input_file, as_gray=True)
    if len(gray_image) == 0:
        raise ValueError(f"Image is empty: {options.input_name}")

    # read color image
    color_image = imageio.imread(options.input_file, as_gray=False)
    if len(color_image) == 0:
        raise ValueError(f"Image is empty: {options.input_name}")

    # binary threshold
    masked_image = thresholding.binary_threshold(gray_image.astype(np.uint8))
    imageio.imwrite(f"{output_prefix}.mask.png", skimage.img_as_uint(masked_image))

    # edge detection
    print(f"Finding edges")
    edges_image = cv2.Canny(color_image, 100, 200)
    cv2.imwrite(f"{output_prefix}.edges.png", edges_image)

    # pad border
    print(f"Padding border")
    border_image = masked_image.copy()
    border_image[[0], :] = [255]
    border_image[[-1], :] = [255]
    border_image[:, [0]] = [255]
    border_image[:, [-1]] = [255]
    cv2.imwrite(f"{output_prefix}.border.png", skimage.img_as_uint(border_image))

    # invert image
    print(f"Inverting")
    inverted_image = (255 - border_image)
    cv2.imwrite(f"{output_prefix}.inverted.png", skimage.img_as_uint(inverted_image))

    # dilate image
    print(f"Dilating")
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=1)
    cv2.imwrite(f"{output_prefix}.dilated.png", dilated_image)

    # close image
    print(f"Closing")
    closed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{output_prefix}.closed.png", closed_image)

    # component labeling
    print(f"Finding connected components")
    num_labels, labels_image, stats, centroids = cv2.connectedComponentsWithStats(dilated_image)
    print(f"Found {num_labels} components")
    label_hue = np.uint8(179 * labels_image / np.max(labels_image))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    cv2.imwrite(f"{output_prefix}.labeled.png", labeled_img)

    # select largest component
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, num_labels):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    largest_comp_image = np.zeros(labels_image.shape)
    largest_comp_image[labels_image == max_label] = 255
    cv2.imwrite(f"{output_prefix}.largest.png", largest_comp_image)


@click.command()
@click.argument('input_file')
@click.option('-o', '--output_directory', required=False, type=str, default='')
def cli(input_file, output_directory):
    start = time.time()
    options = AnalysisOptions(input_file, output_directory)

    print(f"Analyzing image")
    result = process(options)

    print(f"Writing results to file")
    write_results(options, [result])

    duration = ceil((time.time() - start))
    print(f"Finished in {duration} seconds.")


if __name__ == '__main__':
    cli()
