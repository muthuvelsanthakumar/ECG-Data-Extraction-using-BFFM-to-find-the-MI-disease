import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import logging
import os
import matplotlib.pyplot as plt
import pywt  # Import wavelet transform library

# Configure logging
logging.basicConfig(level=logging.INFO)

# Image Preprocessing (unchanged)
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        logging.error(f"File not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error loading image: {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY_INV)
    grid_removed_image = remove_grid_lines(thresholded_image)
    return grid_removed_image

def remove_grid_lines(image):
    kernel = np.ones((3, 3), np.uint8)
    image_dilated = cv2.dilate(image, kernel, iterations=1)
    grid_removed = cv2.erode(image_dilated, kernel, iterations=1)
    return grid_removed

# Lead Segmentation (unchanged)
def segment_leads(image, num_leads=12):
    lead_height = image.shape[0]
    lead_width = image.shape[1] // num_leads
    leads = []
    for i in range(num_leads):
        lead_image = image[:, i*lead_width:(i+1)*lead_width]
        leads.append(lead_image)
    return leads

# Calibration (unchanged)
def calibrate_leads(leads, pixel_to_time=25, pixel_to_voltage=10):
    calibrated_leads = []
    for lead in leads:
        time = np.linspace(0, len(lead) / pixel_to_time, len(lead))
        voltage = np.array([np.mean(lead[i:i+10]) for i in range(0, len(lead), 10)]) * pixel_to_voltage
        calibrated_leads.append((time, voltage))
    return calibrated_leads

# Wavelet Transform Frequency Extraction (instead of FFT)
def calculate_wavelet_frequency(time, voltage):
    """
    Use Wavelet Transform to extract frequency-domain features.
    :param time: Time array for the ECG lead
    :param voltage: Voltage array for the ECG lead
    :return: DataFrame with frequency and magnitude
    """
    # Perform Wavelet Transform
    coeffs, freqs = pywt.cwt(voltage, scales=np.arange(1, 128), wavelet='cmor')
    
    # Average the power across scales (for simplicity, modify as needed)
    magnitude = np.abs(coeffs).mean(axis=1)
    
    df_wavelet = pd.DataFrame({
        'Frequency': freqs,
        'Magnitude': magnitude
    })
    
    return df_wavelet

# Waveform Extraction (unchanged)
def extract_waveform_data(lead_data):
    time, voltage = lead_data
    p_peaks, _ = find_peaks(voltage, height=0.5, distance=30, prominence=0.2)
    qrs_peaks, _ = find_peaks(voltage, height=0.5, distance=10, prominence=0.3)
    t_peaks, _ = find_peaks(voltage, height=0.5, distance=50, prominence=0.2)

    waves = {
        "P_wave": extract_wave_metrics(time, voltage, p_peaks),
        "Q_wave": extract_qrs_metrics(time, voltage, qrs_peaks, "Q"),
        "R_wave": extract_qrs_metrics(time, voltage, qrs_peaks, "R"),
        "S_wave": extract_qrs_metrics(time, voltage, qrs_peaks, "S"),
        "T_wave": extract_wave_metrics(time, voltage, t_peaks),
    }

    return waves

# Frequency Calculation (unchanged)
def extract_wave_metrics(time, voltage, peaks):
    wave_metrics = []
    for peak in peaks:
        start = time[max(0, peak-10)]
        end = time[min(len(time)-1, peak+10)]
        amplitude = voltage[peak]
        wave_metrics.append({
            "start": start,
            "peak": time[peak],
            "end": end,
            "amplitude": amplitude
        })
    return wave_metrics

def extract_qrs_metrics(time, voltage, qrs_peaks, wave_type):
    qrs_metrics = []
    for peak in qrs_peaks:
        if wave_type == "Q":
            q_start = max(0, peak-10)
            q_end = peak
            q_amplitude = voltage[q_start:q_end].min() if len(voltage[q_start:q_end]) > 0 else 0
            q_peak = time[np.argmin(voltage[q_start:q_end]) + q_start]
            qrs_metrics.append({
                "start": time[q_start],
                "peak": q_peak,
                "end": time[q_end],
                "amplitude": q_amplitude
            })
        elif wave_type == "R":
            r_start = peak
            r_end = min(len(time)-1, peak+10)
            r_amplitude = voltage[peak] if len(voltage) > 0 else 0
            r_peak = time[peak]
            qrs_metrics.append({
                "start": time[r_start],
                "peak": r_peak,
                "end": time[r_end],
                "amplitude": r_amplitude
            })
        elif wave_type == "S":
            s_start = peak
            s_end = min(len(time)-1, peak+20)
            s_amplitude = voltage[s_start:s_end].min() if len(voltage[s_start:s_end]) > 0 else 0
            s_peak = time[np.argmin(voltage[s_start:s_end]) + s_start]
            qrs_metrics.append({
                "start": time[s_start],
                "peak": s_peak,
                "end": time[s_end],
                "amplitude": s_amplitude
            })
    return qrs_metrics

# Output Formatting (modified to include wavelet data)
def generate_combined_csv_output(calibrated_leads, num_leads=12):
    all_leads_data = []

    for i in range(num_leads):
        lead_data = calibrated_leads[i]
        waves = extract_waveform_data(lead_data)
        lead_name = f"lead_{i+1}"
        
        for wave_type, wave_metrics in waves.items():
            for metric in wave_metrics:
                try:
                    start_idx = np.where(lead_data[0] == metric["start"])[0][0]
                    end_idx = np.where(lead_data[0] == metric["end"])[0][0]
                    time_segment = lead_data[0][start_idx:end_idx]
                    voltage_segment = lead_data[1][start_idx:end_idx]
                    df_wavelet = calculate_wavelet_frequency(time_segment, voltage_segment)
                    
                    for _, row in df_wavelet.iterrows():
                        all_leads_data.append({
                            "Lead": lead_name,
                            "Wave_Type": wave_type,
                            "Start": metric["start"],
                            "Peak": metric["peak"],
                            "End": metric["end"],
                            "Amplitude": metric["amplitude"],
                            "Frequency": row['Frequency'],
                            "Magnitude": row['Magnitude']
                        })
                except Exception as e:
                    logging.error(f"Error processing lead {lead_name}, wave {wave_type}: {e}")
    
    df = pd.DataFrame(all_leads_data)
    return df

def main(image_path):
    try:
        logging.info("Preprocessing the ECG image...")
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            return
        logging.info("Segmenting the ECG leads...")
        leads = segment_leads(preprocessed_image)
        logging.info("Calibrating leads...")
        calibrated_leads = calibrate_leads(leads)
        logging.info("Generating CSV output...")
        df_output = generate_combined_csv_output(calibrated_leads)
        df_output.to_csv('ecg_combined_wavelet_data.csv', index=False)
        logging.info("ECG waveform data with wavelet frequencies has been written to 'ecg_combined_wavelet_data.csv'.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    image_path = r'D:\Users\A.S.MUTHUVEL\Downloads\DAY 2\task2\Data\MI(1).jpg'  # Path to your ECG report image
    main(image_path)


import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the generated CSV (replace with the correct file path)
df = pd.read_csv(r'D:\Users\A.S.MUTHUVEL\Downloads\Day 4\Using BFFM\ecg_combined_wavelet_data.csv')

# Plotting the data distribution for Frequency and Magnitude
plt.figure(figsize=(10, 6))

# Distribution of Frequency
plt.subplot(2, 1, 1)
plt.hist(df['Frequency'], bins=50, color='skyblue', edgecolor='black')
plt.title('Frequency Distribution of ECG Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count')

# Distribution of Magnitude
plt.subplot(2, 1, 2)
plt.hist(df['Magnitude'], bins=50, color='lightcoral', edgecolor='black')
plt.title('Magnitude Distribution of ECG Data')
plt.xlabel('Magnitude')
plt.ylabel('Count')

# Display the plots
plt.tight_layout()
plt.show()