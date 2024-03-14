# Unsupervised-ML-algorithm-for-galaxy-classification

This is an improved and more robust version of an unsupervised machine learning algorithm originally developed in Hocking et al. 2018 and Martin et al. 2020 involving several clustering and data size reduction techniques to produce a number between 150 and 300 galaxy clusters each containing objects with similar morphological features. 

The code currently takes as inputs individual galaxy images (in .npy format) from the Hyper Supreme-Cam (HSC) survey and catalogs (in .npz format) with galaxy redshifts, stellar masses, HSC star flags. This technique can work with any multiband galaxy survey images as long as the image and catalog formats are respected. In the "examples" folder you can see examples of galaxy morphological clusters that the algorithm produces. Each image contains a representative sample of galaxies (chosen using the distance to the cluster node). The code is run from the "ml_script.py" script.

Needed libraries: scipy, numpy, sklearn, NeuPy (Python version < 3.9)
