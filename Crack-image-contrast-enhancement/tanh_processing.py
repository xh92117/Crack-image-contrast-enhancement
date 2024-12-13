import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

def tanh_hist_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    total_area = gray.shape[0] * gray.shape[1]

    exclude_mask = np.zeros_like(gray, dtype=bool)
    for i in range(1, num_labels):
        region_area = stats[i, cv2.CC_STAT_AREA]
        if region_area > 0.2 * total_area:
            exclude_mask[labels == i] = True

    included_region = gray.copy()
    included_region[exclude_mask] = 0
    excluded_region = gray.copy()
    excluded_region[~exclude_mask] = 0

    included_hist, _ = np.histogram(included_region.flatten(), 256, [0, 256])
    excluded_hist, _ = np.histogram(excluded_region.flatten(), 256, [0, 256])

    smoothed_included_hist = gaussian_filter(included_hist, sigma=2)
    smoothed_excluded_hist = gaussian_filter(excluded_hist, sigma=2)

    included_peaks, _ = find_peaks(smoothed_included_hist, height=0)
    excluded_peaks, _ = find_peaks(smoothed_excluded_hist, height=0)

    included_peak_value = included_peaks[np.argmax(smoothed_included_hist[included_peaks])] if len(included_peaks) > 0 else np.argmax(smoothed_included_hist)
    excluded_peak_value = excluded_peaks[np.argmax(smoothed_excluded_hist[excluded_peaks])] if len(excluded_peaks) > 0 else np.argmax(smoothed_excluded_hist)

    included_area_ratio = np.sum(~exclude_mask) / total_area
    new_peak_value = included_peak_value * included_area_ratio + excluded_peak_value * (1 - included_area_ratio)

    def tanh_function(x):
        if x > new_peak_value:
            return 0.5 * np.tanh(x / new_peak_value) * x + x
        else:
            return -0.5 * np.tanh(x / new_peak_value) * x + x

    mapping = np.zeros(256, dtype=np.float32)
    for i in range(256):
        mapping[i] = tanh_function(i)

    min_val = np.min(gray)
    max_val = np.max(gray)

    equalized = np.interp(gray.flatten(), np.arange(256), mapping).reshape(gray.shape)
    equalized = np.where(exclude_mask, gray, equalized)
    equalized = np.clip(equalized, min_val, max_val).astype(np.uint8)

    return gray, equalized, included_hist, np.histogram(equalized.flatten(), 256, [0, 256])[0]
