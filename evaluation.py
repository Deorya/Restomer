import os
import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from basicsr.metrics.niqe import calculate_niqe
import warnings

# Suppress specific warnings that might not be critical
warnings.filterwarnings("ignore", category=UserWarning, module='skimage')


def evaluate_deraining(ground_truth_folder, derained_folder):
    """
    Evaluates deraining performance, matching 'name.png' with 'name_restored.png'.

    Args:
        ground_truth_folder (str): Path to the folder with ground truth (clean) images.
        derained_folder (str): Path to the folder with the model's output (derained) images.
    """
    psnr_scores, ssim_scores, lpips_scores, niqe_scores = [], [], [], []
    valid_image_count = 0

    # Initialize LPIPS model. Use GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    print(f"Using device: {device}")

    gt_images = sorted(os.listdir(ground_truth_folder))

    if not gt_images:
        print(f"Error: The ground truth folder '{ground_truth_folder}' is empty.")
        return

    for gt_image_name in gt_images:
        print(f"Processing {gt_image_name}...")

        gt_path = os.path.join(ground_truth_folder, gt_image_name)

        # --- MODIFIED LOGIC FOR FILENAME MATCHING ---
        # Derive the restored image name from the ground truth image name
        # e.g., 'rain-001.png' -> 'rain-001_restored.png'
        base_name, extension = os.path.splitext(gt_image_name)
        restored_image_name = f"{base_name}_restored{extension}"
        output_path = os.path.join(derained_folder, restored_image_name)

        if not os.path.exists(output_path):
            print(f"  - Warning: Corresponding restored image '{restored_image_name}' not found. Skipping.")
            continue

        # --- Robust Image Reading ---
        gt_img = cv2.imread(gt_path)
        output_img = cv2.imread(output_path)

        if gt_img is None:
            print(f"  - Error: Failed to read ground truth image: {gt_path}. Skipping.")
            continue
        if output_img is None:
            print(f"  - Error: Failed to read output image: {output_path}. Skipping.")
            continue

        # Ensure images have the same dimensions
        if gt_img.shape != output_img.shape:
            print(f"  - Warning: Mismatched shapes. Resizing output image to match ground truth ({gt_img.shape}).")
            output_img = cv2.resize(output_img, (gt_img.shape[1], gt_img.shape[0]))

        # --- Metric Calculations with Error Handling ---
        try:
            # PSNR
            if np.array_equal(gt_img, output_img):
                psnr_val = np.inf
            else:
                psnr_val = psnr(gt_img, output_img, data_range=255)
            psnr_scores.append(psnr_val)

            # SSIM
            ssim_val = ssim(gt_img, output_img, data_range=255, multichannel=True, channel_axis=2)
            ssim_scores.append(ssim_val)

            # LPIPS
            gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0 * 2 - 1
            output_tensor = torch.from_numpy(output_img).permute(2, 0, 1).unsqueeze(0).float().to(
                device) / 255.0 * 2 - 1
            lpips_val = loss_fn_alex(gt_tensor, output_tensor)
            lpips_scores.append(lpips_val.item())

            # NIQE (calculated on the derained output image)
            niqe_val = calculate_niqe(output_img, crop_border=0)
            niqe_scores.append(niqe_val)

            valid_image_count += 1

        except Exception as e:
            print(f"  - Error calculating metrics for {gt_image_name}: {e}. Skipping.")
            continue

    # --- Averaging and Reporting ---
    print("\n--- Evaluation Results ---")
    if valid_image_count > 0:
        finite_psnr_scores = [s for s in psnr_scores if np.isfinite(s)]
        avg_psnr = np.mean(finite_psnr_scores) if finite_psnr_scores else 0.0

        avg_ssim = np.mean(ssim_scores)
        avg_lpips = np.mean(lpips_scores)
        avg_niqe = np.mean(niqe_scores)

        print(f"Successfully evaluated {valid_image_count} image pairs.")
        print(f"Average PSNR: {avg_psnr:.4f} dB (Higher is better)")
        print(f"Average SSIM: {avg_ssim:.4f} (Higher is better)")
        print(f"Average LPIPS: {avg_lpips:.4f} (Lower is better)")
        print(f"Average NIQE: {avg_niqe:.4f} (Lower is better)")
    else:
        print("No valid image pairs were evaluated. Please check your folder paths and filenames.")


if __name__ == '__main__':
    # --- PLEASE SET YOUR FOLDER PATHS HERE ---
    # 'ground_truth_folder' should contain the clean images (e.g., 'rain-001.png')
    # 'derained_folder' should contain the restored images (e.g., 'rain-001_restored.png')
    ground_truth_folder = 'input_images'
    derained_folder = 'output_images'

    if not os.path.exists(ground_truth_folder):
        print(f"Error: Ground truth folder not found at '{ground_truth_folder}'")
    elif not os.path.exists(derained_folder):
        print(f"Error: Derained images folder not found at '{derained_folder}'")
    else:
        evaluate_deraining(ground_truth_folder, derained_folder)