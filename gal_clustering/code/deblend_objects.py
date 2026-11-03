import os
import glob
import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import label
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def clean_and_square_image(file_path, path_output_dir, threshold_sigma=1.5):
    """
    Loads a stacked .npy file, isolates the central object based on VIS noise,
    masks all other bands, crops it to a perfect square anchored to the ORIGINAL center, 
    and saves it.
    """
    try:
        # Load the 4-band array [VIS, Y, J, H]
        stacked = np.load(file_path)
        vis_band = stacked[0]
        
        # Original geometric center (This is the true center of the galaxy)
        cy, cx = vis_band.shape[0] // 2, vis_band.shape[1] // 2
        
        # 1. Calculate precise background noise on the VIS band
        _, bkg_median, bkg_std = sigma_clipped_stats(vis_band, sigma=3.0, maxiters=5)
        noise_threshold = bkg_median + (threshold_sigma * bkg_std)

        # 2. Create a binary mask of everything above the noise floor
        binary_mask = vis_band > noise_threshold

        # 3. Label all contiguous connected components
        labeled_mask, num_features = label(binary_mask)

        # 4. Identify the central object
        center_label = labeled_mask[cy, cx]

        # Failsafe: If the exact dead-center pixel is below the noise threshold 
        # search a 5x5 box around it for the main object structure.
        if center_label == 0:
            window = labeled_mask[cy-2:cy+3, cx-2:cx+3]
            labels, counts = np.unique(window[window > 0], return_counts=True)
            if len(labels) > 0:
                center_label = labels[np.argmax(counts)]
            else:
                return None # The image is essentially empty/noise

        # Create the final mask containing ONLY the central contiguous object
        central_mask = (labeled_mask == center_label)

        # Apply the mask to ALL 4 bands (sets everything outside the object to 0)
        masked_stacked = stacked * central_mask[np.newaxis, :, :]

        # 5. Find how far the mask extends in every direction
        y_indices, x_indices = np.where(central_mask)
        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()

        # --- NEW LOGIC: Calculate max radius from the ORIGINAL center ---
        radius_y = max(abs(cy - ymin), abs(ymax - cy))
        radius_x = max(abs(cx - xmin), abs(xmax - cx))
        max_radius = max(radius_y, radius_x)

        # The new side length guarantees the original center stays the exact center
        new_size = (2 * max_radius) + 1

        # 6. Extract the square with zero-padding safety
        square_stacked = np.zeros((4, new_size, new_size), dtype=stacked.dtype)

        # Ideal source bounds (may stretch beyond original array bounds)
        src_y0 = cy - max_radius
        src_y1 = cy + max_radius + 1
        src_x0 = cx - max_radius
        src_x1 = cx + max_radius + 1

        # Destination bounds
        dst_y0, dst_y1 = 0, new_size
        dst_x0, dst_x1 = 0, new_size

        # Clamp source bounds to array limits and adjust destination bounds
        if src_y0 < 0:
            dst_y0 -= src_y0
            src_y0 = 0
        if src_y1 > vis_band.shape[0]:
            dst_y1 -= (src_y1 - vis_band.shape[0])
            src_y1 = vis_band.shape[0]
            
        if src_x0 < 0:
            dst_x0 -= src_x0
            src_x0 = 0
        if src_x1 > vis_band.shape[1]:
            dst_x1 -= (src_x1 - vis_band.shape[1])
            src_x1 = vis_band.shape[1]

        # 7. Copy the data into the new centered square array
        square_stacked[:, dst_y0:dst_y1, dst_x0:dst_x1] = masked_stacked[:, src_y0:src_y1, src_x0:src_x1]

        # 8. Save the new clean image
        filename = os.path.basename(file_path)
        new_filepath = os.path.join(path_output_dir, filename.replace('.npy', '_clean.npy'))
        np.save(new_filepath, square_stacked)
        
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def run_cleaning_pipeline(path_input_dir, path_output_dir, threads=16):
    print("STARTING IMAGE ISOLATION PIPELINE...")
    
    os.makedirs(path_output_dir, exist_ok=True)
    
    # Find all stacked .npy files
    search_pattern = os.path.join(path_input_dir, "*_stacked.npy")
    files_to_process = glob.glob(search_pattern)
    
    print(f"Found {len(files_to_process)} images to process.")

    # Setup the worker with fixed arguments
    worker_func = partial(
        clean_and_square_image, 
        path_output_dir=path_output_dir,
        threshold_sigma=1 # <--- Tweak this if it masks out too much or too little of the galaxy edges
    )

    # Run Multiprocessing
    success_count = 0
    with Pool(threads) as pool:
        for result in tqdm(pool.imap_unordered(worker_func, files_to_process), total=len(files_to_process)):
            if result:
                success_count += 1
                
    print(f"PIPELINE FINISHED. Successfully cleaned {success_count}/{len(files_to_process)} images.")


# --- HOW TO RUN IT ---
if __name__ == '__main__':
    # Define your paths
    path_data_dir = "/.../npy_stacked_z015M7/" # Where your current arrays live
    path_clean_dir = "/.../npy_stacked_z015M7_clean/" # Where the new ones will go
    
    run_cleaning_pipeline(path_input_dir=path_data_dir, path_output_dir=path_clean_dir, threads=16)
