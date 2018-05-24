from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from tqdm import tqdm
from tempfile import TemporaryFile
import numba


def loadimages(num_images, image_directory, file_prefix, file_suffix, image_offset, sample_rate):
    """Loads images and corrects for bleaching"""
    first_image_intensity = 0

    print("Loading images.")
    if num_images == 0:
        num_images = setup_load_images(image_directory, file_prefix, file_suffix)
    image_shape = get_image_shape(image_directory, file_prefix, file_suffix)

    tmp_file = TemporaryFile()
    imagelist = np.memmap(tmp_file, mode='w+', dtype=np.int8, shape=(int(num_images), image_shape[1], image_shape[0]))

    for image_index in range(0, num_images):
        image_number = image_index * sample_rate + image_offset
        tmp_image = Image.open('{}{}{:04d}{}'.format(image_directory, file_prefix, image_number, file_suffix))
        tmp_array = np.array(tmp_image.copy())

        # Correct for bleaching by averaging the brightness across all images
        if image_index == 0:
            first_image_intensity = np.mean(tmp_array)
        else:
            tmp_mean = np.mean(tmp_array)
            tmp_array = tmp_array * (first_image_intensity / tmp_mean)

        imagelist[image_index] = tmp_array

    return imagelist, num_images


def get_image_shape(image_directory, file_prefix, file_suffix):
    tmp_img = Image.open(image_directory + file_prefix + "0000" + file_suffix)
    return tmp_img.size


def setup_load_images(image_directory, file_prefix, file_suffix):
    """Globs the image directory to see how many files are there and loads the first image to
    determine its dimensions"""
    file_list = glob(image_directory + file_prefix + "*" + file_suffix)
    num_images = len(file_list)
    if num_images == 0:
        print("No files found.")
        raise FileNotFoundError
    return num_images


@numba.jit
def get_image_difference(image1, image2):
    return image1 - image2


@numba.jit
def get_variance(array):
    return np.var(array)


def main(images_to_load, cutoff, image_directory, file_prefix, file_suffix, num_particles, image_offset, sample_rate):

    # Load the images
    image_list, numimages = loadimages(images_to_load, image_directory, file_prefix, file_suffix, image_offset, sample_rate)
    print("Image Loading complete. Beginning analysis.")
    
    raw_variance = np.zeros(numimages)
    square_variance = np.zeros(numimages)
    samplecount = np.zeros(numimages)
    
    if cutoff > images_to_load or cutoff == 0:
        cutoff = images_to_load

    loop_counter = 0
    pbar = tqdm(total=int(((cutoff - 1) ** 2 + (cutoff - 1)) / 2 + (numimages - cutoff) * cutoff))
    
    for image1 in range(cutoff):
        for image2 in range(image1 + 1, numimages):
            image_difference = get_image_difference(image_list[image2], image_list[image1])
            image_variance = get_variance(image_difference)
            raw_variance[image2-image1] += image_variance
            square_variance[image2-image1] += image_variance ** 2
            samplecount[image2-image1] += 1
            loop_counter += 1
        pbar.update(loop_counter)
        loop_counter = 0
    pbar.close()

    chi_squared = np.zeros((numimages, 2))
    chi_squared[:, 0] = np.arange(0, numimages, 1) * sample_rate
    chi_squared[:, 1] = do_normalisation(num_particles, numimages, raw_variance, samplecount, square_variance)

    # plt.semilogx(chi_squared[0:], chi_squared[1:], marker="*")
    # plt.show()

    np.savetxt("chi4_offset{:04d}_sample_rate{:d}.txt".format(image_offset, sample_rate), chi_squared, fmt=['%d', '%1.5f'])


def do_normalisation(num_particles, numimages, raw_variance, samplecount, square_variance):
    raw_variance[1:] /= samplecount[1:]
    square_variance[1:] /= samplecount[1:]
    asymptotic_variance = raw_variance[int(numimages * 0.8)]
    variance_squared = raw_variance ** 2
    chi_squared = (square_variance - variance_squared) * num_particles / (asymptotic_variance ** 2)
    return chi_squared


if __name__ == '__main__':
    images_to_load = 1000
    cutoff = 0
    image_directory = "F:/sample DH/Pastore/Raw Images/images/"
    file_prefix = "vacq00_vf071"
    file_suffix = ".tif"
    num_particles = 600
    offsets_list = [0]
    sample_rate = 10

    for offset in offsets_list:
        main(images_to_load, cutoff, image_directory, file_prefix, file_suffix, num_particles, offset, sample_rate)
