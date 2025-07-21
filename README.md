# OCT Retinal Analysis

This project provides tools and models for analyzing Optical Coherence Tomography (OCT) retinal images to assist in the detection of retinal diseases such as CNV, DME, DRUSEN, and NORMAL.

## Dataset Information

This project uses a large, labeled Optical Coherence Tomography (OCT) image dataset for retinal disease classification. The dataset contains four classes: CNV, DME, DRUSEN, and NORMAL.

- **Source:** Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3, doi: [10.17632/rscbjbr9sj.3](https://data.mendeley.com/datasets/rscbjbr9sj/3)
- **Version Used:** Version 1 (OCT images only)
- **Classes:** CNV, DME, DRUSEN, NORMAL
- **Original Split:** 108,309 train images, 1,000 test images
- **Custom Split:** All 109,309 images were combined and re-split into train, validation, and test sets for this project.

## Project Structure
- `eye_api.py`: Main API for image analysis and prediction.
- `geolocation_component.py`: Geolocation utilities for associating image data with location.
- `human-eye.ipynb`: Jupyter notebook for exploratory analysis and model evaluation.
- `Trained_Model.h5` / `Trained_Model.keras`: Pre-trained deep learning models for retinal image classification.
- `Training_history.pkl`: Training history for model performance tracking.
- `test/`: Contains test images organized by disease category (CNV, DME, DRUSEN, NORMAL).

## Getting Started
1. Clone the repository and ensure you have Python 3.11+ installed.
2. Install required packages (see notebook or scripts for dependencies).
3. Use `eye_api.py` to run predictions on new OCT images.
4. Explore `human-eye.ipynb` for analysis and visualization.

## Installation
1. Clone this repository:
   ```powershell
   git clone <repo-url>
   cd oct-retinal-analysis
   ```
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage Example
```bash
python eye_api.py --image_path test/CNV/CNV-1016042-122.jpeg
```

## License
This project is for educational and research purposes.

## Contact
For questions, please contact the project maintainer.
#
# Features
- Deep learning-based classification of OCT retinal images
- Supports detection of CNV, DME, DRUSEN, and NORMAL classes
- Pre-trained models included for quick inference
- Jupyter notebook for model training, evaluation, and visualization
- Easy-to-use API for batch and single image prediction
- Geolocation component for associating images with location data

# Example Workflow
1. Prepare your OCT image(s) in the appropriate test folder (e.g., `test/CNV/`).
2. Run the API script to get predictions:
   ```powershell
   python eye_api.py --image_path test/CNV/CNV-1016042-122.jpeg
   ```
3. Open `human-eye.ipynb` to explore model training, evaluation, and visualization.

# Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

# Troubleshooting
- Ensure all dependencies are installed using `requirements.txt`.
- Use Python 3.11 or newer for best compatibility.
- If you encounter issues with TensorFlow, check your system's CUDA and cuDNN setup if using GPU.