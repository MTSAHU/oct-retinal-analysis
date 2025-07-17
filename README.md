# OCT Retinal Analysis

This project provides tools and models for analyzing Optical Coherence Tomography (OCT) retinal images to assist in the detection of retinal diseases such as CNV, DME, DRUSEN, and NORMAL.

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