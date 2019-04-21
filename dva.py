from PIL import Image
import numpy as np
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
    if len(image_shape) == 3:
        imagelist = np.memmap(tmp_file, mode='w+', dtype=np.int8, shape=(int(num_images), image_shape[0], image_shape[2], image_shape[1]))
    else:
        imagelist = np.memmap(tmp_file, mode='w+', dtype=np.int8, shape=(int(num_images), image_shape[1], image_shape[0]))
    for image_index in range(num_images):
        image_number = image_index * sample_rate + image_offset
        image_path = '{}{}{:04d}{}'.format(image_directory, file_prefix, image_number, file_suffix)
        tmp_array = load_image_from_file(image_path)

        # Correct for bleaching by averaging the brightness across all images
        if image_index == 0:
            first_image_intensity = np.mean(tmp_array)
        else:
            tmp_mean = np.mean(tmp_array)
            tmp_array = tmp_array * (first_image_intensity / tmp_mean)

        imagelist[image_index] = tmp_array

    return imagelist, num_images


def load_image_from_file(image_path):
    if file_suffix == ".png" or file_suffix == ".tif":
        tmp_image = Image.open(image_path)
    elif file_suffix == ".npy":
        tmp_image = np.load(image_path)
    else:
        print("File type not understood.")
        raise TypeError
    tmp_array = np.array(tmp_image.copy())
    return tmp_array


def get_image_shape(image_directory, file_prefix, file_suffix):
    image_path = image_directory + file_prefix + "0000" + file_suffix
    tmp_img = load_image_from_file(image_path)
    return tmp_img.shape


def setup_load_images(image_directory, file_prefix, file_suffix):
    """Globs the image directory to see how many files there are"""
    file_list = glob(image_directory + file_prefix + "*" + file_suffix)
    num_images = len(file_list)
    if num_images == 0:
        print("No files found.")
        raise FileNotFoundError
    return num_images


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
    
    if cutoff > numimages or cutoff == 0:
        cutoff = numimages

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

    np.savetxt("chi4_offset{:04d}_sample_rate{:d}.txt".format(image_offset, sample_rate), chi_squared, fmt=['%d', '%1.5f'])


def do_normalisation(num_particles, numimages, raw_variance, samplecount, square_variance):
    raw_variance[1:] /= samplecount[1:]
    square_variance[1:] /= samplecount[1:]
    asymptotic_variance = raw_variance[int(numimages * 0.8)]
    variance_squared = raw_variance ** 2
    chi_squared = (square_variance - variance_squared) * num_particles / (asymptotic_variance ** 2)
    return chi_squared


if __name__ == '__main__':
    images_to_load = 64
    cutoff = 0
    image_directory = "F:/sample DH/James/10_5_16-vf-58/pickle_frames/"
    file_prefix = "i_"
    file_suffix = ".npy"
    num_particles = 6000
    offsets_list = [0]
    sample_rate = 1

    for offset in offsets_list:
        main(images_to_load, cutoff, image_directory, file_prefix, file_suffix, num_particles, offset, sample_rate)
