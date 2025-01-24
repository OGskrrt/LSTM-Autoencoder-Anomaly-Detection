## Project Overview

This project demonstrates how to:
1. Read Heart Rate (HR) signals from WFDB-format files in a driver stress dataset.
2. Combine the HR signals from multiple drivers into a single Pandas DataFrame.
3. Perform scaling (StandardScaler/MinMaxScaler) on these signals.
4. Create an LSTM-based Autoencoder model to learn a representation of these signals.
5. Detect anomalous segments (potentially high-stress signals) by computing reconstruction errors.

> **Note on Training Limitations**  
> Due to hardware constraints, the autoencoder cannot be trained for many epochs or with larger batch sizes. Consequently, the model may not achieve extremely high accuracy. However, the current setup serves as a proof of concept for detecting anomalies (potential signs of driver stress) based on reconstruction errors.

## Main Steps

1. **Reading WFDB Files**  
   We iterate over all `.dat` files in the specified directory, extract the "HR" channel if present, and store it in a combined DataFrame (`all_hr_signals`).

2. **Scaling**  
   We optionally apply standard or min-max scaling to normalize the signals.

3. **Batch Preparation**  
   The `BatchLoader` class segments each driver’s data into smaller time-based chunks, normalizes each chunk, then windows them (e.g., 50 samples per window) for model input.

4. **Autoencoder Training**  
   A simple LSTM autoencoder is trained (one window size in, one window size out). Training loss (MSE) is logged each epoch.  
   - **Hardware Constraints**: Limited GPU/CPU resources can restrict the number of training epochs and the complexity of the model, potentially reducing final accuracy.

5. **Anomaly Detection**  
   Reconstruction error (MAE) is measured for each window. Windows with an MAE above a chosen threshold are flagged as anomalies. Each driver’s first 3,600 samples (1 hour, for instance) are analyzed, plotting anomalies on the raw HR signal.

## Usage
- **Install Requirements**: The code uses `wfdb`, `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`.  
- **Paths and Parameters**: Update `input_dir` to point to your `.dat` files. Adjust `window_size`, `batch_size`, and threshold values as needed.  
- **Run**: Execute the cells in sequence. The final plots display anomalous segments for each driver.

## Dataset Information

### Context
This dataset contains multiparameter recordings from healthy volunteers driving in and around Boston, Massachusetts, as part of a study on automated stress recognition. It includes ECG, EMG (right trapezius), GSR, and respiration signals, but here we specifically focus on the "HR" (Heart Rate) channel for anomaly detection.

### Content
The signals were collected for the purpose of investigating how to automatically detect stress in real-world driving tasks. The dataset features various channels; in this project, we concentrate on the HR channel to illustrate a basic anomaly detection workflow using an LSTM autoencoder.

### Acknowledgements
These data come from:
- Healey JA, Picard RW. *Detecting stress during real-world driving tasks using physiological sensors.* IEEE Transactions in Intelligent Transportation Systems, 6(2):156-166 (June 2005).

### Dataset Link
[Kaggle: Stress Recognition in Automobile Drivers](https://www.kaggle.com/datasets/bjoernjostein/stress-recognition-in-automobile-drivers)
