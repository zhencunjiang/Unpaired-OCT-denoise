import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def resize_image(image, target_size):
    resized = cv2.resize(image, target_size)
    return resized


def calculate_psnr(image1, image2):
    psnr = peak_signal_noise_ratio(image1, image2)
    return psnr


def calculate_ssim(image1, image2):
    ssim = structural_similarity(image1, image2, multichannel=True)
    return ssim


def calculate_rmse(image1, image2):
    rmse = np.sqrt(np.mean((image1 - image2) ** 2))
    return rmse

def calculate_epi(original_matrix, processed_matrix):
    original_edges = np.abs(np.gradient(original_matrix))
    processed_edges = np.abs(np.gradient(processed_matrix))
    numerator = np.sum(np.bitwise_and(original_edges > 0, processed_edges > 0))
    denominator = np.sum(original_edges > 0)
    epi = numerator / denominator

    return epi
def compare_images(folder1_path, folder2_path):
    file_list1 = os.listdir(folder1_path)
    file_list2 = os.listdir(folder2_path)

    psnr_scores = []
    ssim_scores = []
    rmse_scores = []
    epi_scores=[]
    for i in range(len(file_list1)):
        file1_path = os.path.join(folder1_path, file_list1[i])
        file2_path = os.path.join(folder2_path, file_list2[i])

        image1 = cv2.imread(file1_path)/255
        image2 = cv2.imread(file2_path)/255

        psnr = calculate_psnr(image1, image2)
        ssim = calculate_ssim(image1, image2)
        rmse = calculate_rmse(image1, image2)
        epi=calculate_epi(image1,image2)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        rmse_scores.append(rmse)
        epi_scores.append(epi)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_epi=np.mean(epi_scores)

    return avg_psnr, avg_ssim, avg_rmse



folder1_path = '/home/ps/zhencunjiang/pycharm_project_773/OCT_DDPM-main/noise_diff/duke17'
folder2_path = '/home/ps/zhencunjiang/pycharm_project_773/OCT_DDPM-main/noise_diff_clean/duke17'


psnr, ssim, rmse= compare_images(folder1_path, folder2_path)

print("Average PSNR:", psnr)
print("Average SSIM:", ssim)
print("Average RMSE:", rmse)
