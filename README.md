# Road Lane Line Detection with FCN Architecture

## Project Overview

This project uses an FCN architecture for detecting road lane lines. The dataset used is the pre-processed TuSimple dataset, available on Kaggle. The model employs a ResNet50 encoder and a smaller Fully Convolutional Network (FCN) decoder to achieve semantic segmentation of road lane lines.

## Features

- **Preprocessed Dataset**: TuSimple dataset, which includes road images and corresponding lane masks.
- **FCN Architecture**: A combination of ResNet50 as the encoder and a custom decoder.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, and Intersection over Union (IoU).
- **Output**: Predicted lane masks with visualizations saved locally.

---

## Installation Instructions

### macOS

To run the project on macOS, ensure you have the following prerequisites:

1. **Python Version**: 3.10 or 3.11 (Recommended: 3.10.6).
2. **Install Dependencies**:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib pandas opencv-python scikit-learn tensorflow-macos tensorflow-metal
```

### Windows

To run the project on Windows:

1. **Python Version**: 3.10 or 3.11 (Recommended: 3.10.6).
2. **Install Dependencies**:

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy matplotlib pandas opencv-python scikit-learn tensorflow
```

### Google Colab

To run the project on Google Colab:

1. **Upload Kaggle API Key**: Upload `kaggle.json` to the Colab environment.
2. **Install Dependencies**:

```python
!pip install -q kaggle tensorflow numpy matplotlib pandas opencv-python scikit-learn
```

3. **Download Dataset**:

```python
from google.colab import files

files.upload()
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d hikmatullahmohammadi/tusimple-preprocessed

import zipfile
with zipfile.ZipFile("tusimple-preprocessed.zip", "r") as zip_ref:
    zip_ref.extractall("tusimple-dataset-preprocessed")
```

---

## How to Run the Project

1. Clone the repository or download the code files.
2. Ensure the necessary dependencies are installed (as described above).
3. Place the dataset in the correct folder structure, or follow the Google Colab instructions to download and extract the dataset.
4. Run the main script or notebook.

```bash
python main.py  # If running as a script
# or
jupyter notebook Lane_Detection.ipynb  # If using a notebook
```

---

## Notes

- macOS users should use TensorFlow Metal to leverage GPU acceleration.
- Windows users can directly install TensorFlow for both CPU and GPU support.
- Google Colab provides a free GPU runtime, making it a convenient option for training models.

---

## Contribution Guidelines

Feel free to submit issues and pull requests for improvements or bug fixes.
