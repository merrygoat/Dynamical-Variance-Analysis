from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numba
%matplotlib inline


def loadimages(num_images, image_directory, file_prefix, file_suffix):
    print("Loading images.")
    imagelist = []

    if num_images == 0:
        file_list = glob(image_directory + file_prefix + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    for i in range(num_files):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(i) + file_suffix)
        tmp_array = np.array(tmp_image.copy(), dtype=np.int16)

        # Correct for bleaching by averaging the brightness across all images
        if i == 0:
            first_mean = np.mean(tmp_array)
        else:
            tmp_mean = np.mean(tmp_array)
            tmp_array = tmp_array * (first_mean / tmp_mean)

        imagelist.append(tmp_array)

    print("Image Loading complete. Beginning analysis.")
    return imagelist, num_files

	
@numba.jit
def imagediff(image1, image2):
    return image1 - image2

	
@numba.jit
def variance(array):
    return np.var(array)

	
def dynamical_heterogenity(images_to_load, cutoff, image_directory, file_prefix, file_suffix)

	image_list, numimages = loadimages(images_to_load, image_directory, file_prefix, file_suffix)
	print(image_list[0].nbytes)
	bins = np.arange(0, numimages, 1)
	
	raw_variance = np.zeros((numimages))
	square_variance = np.zeros((numimages))
	samplecount = np.zeros(numimages)
	
	loop_counter = 0
	pbar = tqdm(total=int(((cutoff - 1) ** 2 + (cutoff - 1)) / 2 + (numimages - cutoff) * cutoff))
	
	for image1 in range(cutoff):
		for image2 in range(image1 + 1, numimages):
			image_variance = variance(imagediff(image_list[image2], image_list[image1]))
			raw_variance[image2-image1] += image_variance
			square_variance[image2-image1] += image_variance ** 2
			samplecount[image2-image1] += 1
			loop_counter += 1
		pbar.update(loop_counter)
		loop_counter = 0
	
	long_time_plateau = raw_variance[int(numimages*0.8)]
	# Normalisation
	raw_variance[1:] /= samplecount[1:]
	square_variance[1:] /= samplecount[1:]
	#normed_variance = 1-(raw_variance/long_time_plateau)
	#plt.semilogx(bins[1:], normed_variance[1:])
	
	variance_squared = raw_variance ** 2
	
	num_particles = 590
	asymptotic_variance = 940
	
	chi_squared = (square_variance - variance_squared)*num_particles/(asymptotic_variance**2)
	plt.semilogx(bins[1:], chi_squared[1:], marker="*")
	plt.show()