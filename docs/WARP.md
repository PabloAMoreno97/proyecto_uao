# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a medical diagnostic tool for rapid pneumonia detection in chest X-ray images. The application uses deep learning to classify radiographic images into three categories:

1. **Bacterial Pneumonia** (`bacteriana`)
2. **Viral Pneumonia** (`viral`) 
3. **No Pneumonia** (`normal`)

The tool features a Tkinter-based GUI that loads DICOM or standard image files, runs inference using a pre-trained CNN model, and provides visual explanations via Grad-CAM heatmaps.

## Core Architecture

### Modular Pipeline Design
The application follows a clear separation of concerns with each module handling a specific stage:

```
detector_neumonia.py (GUI) -> integrator.py -> [read_img.py, preprocess_img.py, load_model.py, grad_cam.py]
```

### Key Components

- **`detector_neumonia.py`**: Main GUI application using Tkinter. Contains the `App` class that manages the complete user interface, file dialogs, and button interactions.

- **`integrator.py`**: Central orchestrator that coordinates the prediction pipeline. The `predict()` function calls preprocessing, model inference, and Grad-CAM generation in sequence.

- **`read_img.py`**: Image I/O module with functions for DICOM (`read_dicom_file()`) and JPEG (`read_jpg_file()`) format handling. **Note**: Missing `import pydicom as dicom` - this needs to be added.

- **`preprocess_img.py`**: Image preprocessing pipeline that resizes to 512×512, converts to grayscale, applies CLAHE contrast enhancement, normalizes to [0,1], and formats as model input tensor.

- **`load_model.py`**: Model loading utility that loads the pre-trained CNN from `modelo/conv_MLP_84.h5`. Currently has basic error handling.

- **`grad_cam.py`**: Gradient-weighted Class Activation Mapping implementation targeting the `conv10_thisone` layer (64 filters) for generating explanation heatmaps.

### Neural Network Architecture
The CNN model is based on research by F. Pasa et al. featuring:
- 5 convolutional blocks with skip connections (16, 32, 48, 64, 80 filters)
- 3×3 convolutions throughout
- Max pooling after each conv block
- Three fully-connected layers (1024, 1024, 3 neurons)
- Dropout regularization at 20% in blocks 4, 5 and first Dense layer

## Development Commands

### Environment Setup

#### Using environment.yml (Recommended)
```bash
# Create environment from YAML file with all dependencies and versions
conda env create -f environment.yml
conda activate pneumonia-detector
```

#### Alternative setup
```bash
# Create conda environment with TensorFlow
conda create -n tf tensorflow
conda activate tf

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the GUI application
python detector_neumonia.py
```

### Docker Usage
```bash
# Build container
docker build -t pneumonia-detector .

# Note: GUI applications in Docker require X11 forwarding on Linux/macOS
```

### Development Testing
```bash
# Test individual modules (no formal test suite exists)
python -c "from read_img import read_dicom_file; print('Image reader works')"
python -c "from load_model import model_fun; print('Model loader works')"
python -c "from integrator import predict; print('Integration module works')"
```

## Important Technical Details

### Model Dependencies
- The application expects a model file at `modelo/conv_MLP_84.h5`
- The Grad-CAM implementation specifically targets the `conv10_thisone` layer
- TensorFlow v1 compatibility mode is enabled for the model

### Image Processing Pipeline
1. **Input**: DICOM (.dcm), JPEG, JPG, or PNG files
2. **Preprocessing**: 512×512 resize → grayscale → CLAHE → normalization → tensor format
3. **Inference**: CNN prediction with confidence score
4. **Visualization**: Grad-CAM heatmap overlay at 80% transparency

### Known Issues
- `read_img.py` is missing `import pydicom as dicom` for DICOM file support
- No formal test suite or validation scripts
- Hard-coded layer name `conv10_thisone` creates tight coupling
- GUI uses absolute positioning making it non-responsive

### Data Flow
```
User selects image → read_img.py → preprocess_img.py → load_model.py (inference) → grad_cam.py → GUI display
```

### Output Files
- **historial.csv**: Patient results with ID, diagnosis, and confidence
- **ReporteN.pdf**: Screenshot-based PDF reports (where N is incremented)

## Code Patterns

### Error Handling
- Basic try-catch in model loading
- GUI confirmation dialogs for destructive operations
- File dialog validation for supported formats

### State Management
- GUI state managed through Tkinter variables (`StringVar`)
- Model loaded on each prediction (no caching)
- Results stored in instance variables for CSV/PDF export

### Dependencies Management
- TensorFlow/Keras for deep learning inference
- OpenCV for image processing and Grad-CAM visualization
- PIL/Pillow for image format conversion
- pydicom for medical image format support
- pandas for CSV operations
