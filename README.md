# Unsupervised Machine Learning (UML) for Galaxy Morphological Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MNRAS](https://img.shields.io/badge/MNRAS-Lazar%20et%20al.%202023-darkred)](https://doi.org/10.1093/mnras/stad223)

An end-to-end, multi-band **Unsupervised Machine Learning (UML)** framework for automated galaxy morphological classification and anomaly discovery in wide-field astronomical surveys. 

Building upon the patch-based manifold learning methodology originally introduced by [Hocking et al. (2018)](https://doi.org/10.1093/mnras/stx2651) and [Martin et al. (2020)](https://doi.org/10.1093/mnras/stz3006), this pipeline combines local spatial-frequency feature extraction with global structural metrics, dimensionality reduction, and hierarchical/partitional clustering. It autonomously groups massive, unlabelled galaxy datasets into **~150–500 high-purity morphological micro-clusters**—ranging from grand-design spirals, barred disks, and early-type spheroids to dwarf galaxies, low-surface-brightness (LSB) systems, and interacting/lensing anomalies—requiring **zero pre-labelled training data**.

The codebase has been tested, refactored, and validated on **Hyper Suprime-Cam (HSC-SSP; Rubin-LSST-like)** and **Euclid** imaging surveys (see [Lazar et al. 2023](https://doi.org/10.1093/mnras/stad223)).

## Key Features & Pipeline Architecture

1. **Multi-Scale Patch Extraction & 2D Fourier Power Spectra:** 
   Extracts multi-band sub-patches across each galaxy cutout and computes radially binned 2D Fast Fourier Transform (FFT) power spectra weighted by dual spatial-frequency Gaussian kernels, alongside local colour and structural gradients.
2. **Global Structural & Lensing Vector:** 
   Complements local patch textures with a 10-parameter galaxy-wide structural vector, including Concentration ($C$), Asymmetry ($A$), Clumpiness ($S$), $M_{20}$, radially binned $m=2$ Fourier modes ($m_{2,\text{in}}, m_{2,\text{out}}$), boxiness/diskiness ($a_4$), axis ratio ($b/a$), Centroid Shift ($D_0$), Ring Index ($R_{\text{I}}$), and an Inner/Outer Colour Gradient Index.
3. **Morphological Codebook Generation:** 
   Compresses millions to billions of extracted sub-patches into a discrete visual codebook using `MiniBatchKMeans` and **Hierarchical Agglomerative Clustering (HAC)**.
4. **TF-IDF Weighting & Automated SVD Knee Detection:** 
   Transforms galaxy patch-frequency histograms via sublinear TF-IDF weighting and $L_2$ normalization, then automatically determines the intrinsic structural noise floor using `TruncatedSVD` variance knee-finding.
5. **Automated Silhouette-Score Sweep:** 
   Executes an automated Silhouette-score optimization sweep across K-Means clusterings to partition galaxies into an optimal number (~150–500) of high-purity morphological micro-clusters.

---

## Input Data Format

The pipeline accepts the following inputs:
* **Galaxy Cutout Images (`.npy` format):** Monochrome (single-band) or multi-band postage-stamp arrays of galaxies.
* **Source Catalogues (`.npz` or `.csv` format):** Pre-selection metadata tables containing object IDs, coordinates, photometric/spectroscopic redshifts ($z$), stellar masses ($M_\star$), and resolution flags.

---

## Repository Structure & Quick Start

* `ml_script.py` — Master execution script that runs the end-to-end UML pipeline (patch extraction, codebook learning, TF-IDF/SVD dimensionality reduction, galaxy-level clustering, and diagnostic visualization).
* `examples/` — Contains representative visual montages of morphological micro-clusters generated autonomously by the algorithm on **HSC-SSP (LSST-like)** and **Euclid** imaging. Each montage displays representative galaxies ranked by their Euclidean distance to the cluster centroid in feature space.

### Running the Pipeline

1. **Clone the repository:**

       git clone https://github.com/Ilin01/Unsupervised-ML-algorithm-for-galaxy-classification.git
       cd Unsupervised-ML-algorithm-for-galaxy-classification

2. **Install dependencies:**
   Ensure you have a standard scientific Python environment (`numpy`, `scipy`, `scikit-learn`, `astropy`, `pandas`, `matplotlib`, `sep`).

3. **Execute the main pipeline:**
   Configure your input cutout and catalogue paths inside `ml_script.py` and run:

       python ml_script.py

---

## Citations & References

If you use or adapt this software for your research, please cite the following papers:

* **Lazar, I., et al. (2023)**, *MNRAS*, 520, 2109 — [The morphological mix of galaxies in the low-redshift Universe](https://doi.org/10.1093/mnras/stad223)
* **Martin, G., et al. (2020)**, *MNRAS*, 491, 1408 — [Galaxy morphological classification in deep-wide surveys via unsupervised machine learning](https://doi.org/10.1093/mnras/stz3006)
* **Hocking, A., et al. (2018)**, *MNRAS*, 473, 1108 — [An automatic taxonomy of galaxy morphology using unsupervised machine learning](https://doi.org/10.1093/mnras/stx2651)

---

## Contact

**Dr. Ilin Lazar**  
GitHub: [@Ilin01](https://github.com/Ilin01)