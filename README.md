# Dynamic Variance Analysis

Based on method in Pastore, R. et al. Differential Variance Analysis: a direct method to quantify and visualize dynamic heterogeneities, Sci Rep. 2017; 7: 43496.

Main routine is called dynamical_heterogeneity. Example IPython Notebook shows usage.

Parameters are:
	
* images_to_load  - Number of timesteps to load from disk. Set to 0 for all available images.
* cutoff          - Maximum averaging for each timestep. Set to 0 for analysis of all images.
* image_directory -	Directory path of images to analyse, end with slash
* file_prefix     - Prefix for file. Program assumes files are numbered sequentially from 0 with 4 digits i.e. image_0000.png, image_0001.png...
* file_suffix     - Image file type, ".png" or ".tif" are know to work. Other formats untested.
* num_particles   - Average number of particles in the images, used for normalisation.

Required libraries: numpy, matplotlib, numba, tqdm, pillow
For the example file: IPython Notebook

Beware: This code nearly works though the normalisation isn't quite right yet.