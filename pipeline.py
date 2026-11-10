import os
import glob
import cv2
import numpy as np
np.random.seed(111)

DIR_OG = "data/original"
DIR_DEG = "data/degraded"
DIR_ENH = "data/enhanced"
DIR_FIG = "results/figures"

# create degraded, enhanced and figures folders 
for d in [DIR_DEG, DIR_ENH, DIR_FIG, "results"]:
    os.makedirs(d, exist_ok=True)

# mse calculus
def mse_func(img1, img2):
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

# psnr calculus
def psnr_func(img1, img2):
    mse = mse_func(img1, img2)
    if mse == 0:
        return 100.0
    return round(10.0 * np.log10((255.0 ** 2) / mse), 2)

# applying systetic degradation, lowering contrast and adding noise
def degrade_func(img):
    low_contra = (img.astype(np.float32) - 128.0) * 0.45 + 128.0
    noise = np.random.normal(0, 12.0, img.shape)
    return np.clip(low_contra + noise, 0, 255).astype(np.uint8)

# enhanced evaluation (using linear contrast stretching)
def linear_stretching(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB) # converting to LAB and working only on L channel
    L, a, b = cv2.split(lab)
    min_v, max_v = np.percentile(L, (1.0, 99.0))
    if max_v > min_v:
        L_str = np.clip((L.astype(np.float32) - min_v) * (255.0 / (max_v - min_v)), 0, 255).astype(np.uint8)
    else:
        L_str = L
    return cv2.cvtColor(cv2.merge([L_str, a, b]), cv2.COLOR_LAB2BGR) # revert back to BGR 

# hist euqlization
def hist_equalization(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB) # converting to LAB and working only on L channel
    L, a, b = cv2.split(lab)
    L_eq = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L_eq, a, b]), cv2.COLOR_LAB2BGR) #revert back to BGR 

# bilateral filter
def bilateral_denoising(img_bgr):
    return cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)

# pipeline execution 
images = sorted(glob.glob(os.path.join(DIR_OG, "*.*"))) # taking the iamges from the 'original' folder
if not images:
    print(f"select from 3 to 5 images in {DIR_OG}")
    exit()

with open("results/metrics.csv", "w") as f_out: # CSV file to store the evaluation results
    f_out.write("Image,Method,PSNR_dB,MSE\n")
    for path in images:
        nome = os.path.splitext(os.path.basename(path))[0]
        orig = cv2.imread(path)
        if orig is None:
            continue
        deg = degrade_func(orig) # applying syntetic degradation and saving the result
        cv2.imwrite(os.path.join(DIR_DEG, f"{nome}_degraded.png"), deg)
        # applying the rest of the enhanced methods
        results = {
            "Baseline": deg, # no enhancement
            "Linear_Stretching": linear_stretching(deg),
            "Hist_Equalization": hist_equalization(deg),
            "Bilateral_Filter": bilateral_denoising(deg)
        }
        # saves the outputs and calculates the metrics for each image
        for name_method, img_res in results.items():
            cv2.imwrite(os.path.join(DIR_ENH, f"{nome}_{name_method}.png"), img_res) # saving the enhanced image to compare it using MSE and PSNR metrics
            # calculating evaluation metrics
            mse = round(mse_func(orig, img_res), 2)
            psnr = psnr_func(orig, img_res)
            f_out.write(f"{nome},{name_method},{psnr},{mse}\n") # write the result to the CSV file
            print(f"[{nome}] {name_method:<20} -> PSNR: {psnr:<8} dB MSE: {mse}") # writing the result also on the console log
            # creating side-by-side comparison with original, degraded and enhanced image
            side_by_side = np.hstack([orig, deg, img_res])
            cv2.imwrite(os.path.join(DIR_FIG, f"{nome}_{name_method}_side_by_side.png"), side_by_side)