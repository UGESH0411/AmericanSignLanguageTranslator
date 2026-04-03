# American Sign Language Translator

A real-time sign language translator that recognizes American Sign Language (ASL) letters using deep learning and computer vision.

## Features

- Real-time hand detection using MediaPipe
- Deep neural network for letter classification
- Live video prediction with confidence scoring
- GUI interface with character accumulation
- Text-to-speech output for recognized words
- Support for letters: A, B, C, D, E, F, L

## Requirements

- Python 3.7+
- TensorFlow/Keras
- OpenCV
- MediaPipe
- Tkinter
- pyttsx3
- Pillow
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/UGESH0411/AmericanSignLanguageTranslator.git
cd AmericanSignLanguageTranslator
```

2. Install dependencies:
```bash
pip install -r Scripts/requirement.txt
```

3. Ensure you have a trained model at `Model/trained_model.h5`

## Usage

### Test the Model

Run the real-time translator with your webcam:
```bash
python Scripts/Test_model.py
```

**Controls:**
- **Reset**: Clear all detected letters and start fresh
- **Clear**: Clear the last detected letter
- **Speech**: Read out the formed word using text-to-speech
- **Exit**: Close the application

### Train the Model

To train with your own dataset:
```bash
python Scripts/Train_model.py
```

### Collect Dataset

To collect new training data:
```bash
python Scripts/DatasetCollector.py
```

## Dataset Structure

```
Data/
├── Train/         # Training images
├── Test/          # Test images
└── Validation/    # Validation images
    ├── A/
    ├── B/
    ├── C/
    ├── D/
    ├── E/
    ├── F/
    └── L/
```

Each letter folder contains corresponding sign images (128x128 pixels recommended).

## Model Details

- **Input**: 128x128 RGB hand images
- **Output**: Probability distribution over 7 letter classes
- **Confidence Threshold**: 0.57 (adjustable in `Test_model.py`)
- **Stability Threshold**: 5 consecutive frames for letter confirmation
- **Format**: TensorFlow/Keras (.h5)

## Architecture

1. **Hand Detection**: MediaPipe Hands solution detects hand landmarks
2. **Preprocessing**: Crops largest hand, resizes to 128x128, normalizes
3. **Classification**: Deep CNN predicts letter from hand image
4. **Temporal Filtering**: Confirms predictions over multiple frames to reduce false positives
5. **Output**: Displays letter and accumulates to form words

## Configuration

Key parameters in `Test_model.py`:
- `min_detection_confidence`: Hand detection threshold (default: 0.7)
- `min_tracking_confidence`: Hand tracking threshold (default: 0.7)
- `confidence`: Minimum prediction confidence (default: 0.57)
- `stable_threshold`: Frames needed to confirm letter (default: 5)

## Troubleshooting

- **Webcam not found**: Ensure your camera is connected and not in use by another application
- **Model loading error**: Verify `Model/trained_model.h5` exists and is a valid Keras model
- **Poor recognition**: Ensure good lighting, clear hand position, and matching training data
- **Speech not working**: Check system audio settings and pyttsx3 installation

## Project Structure

```
SignLanguageTranslator/
├── Data/              # Dataset folders (Train, Test, Validation)
├── Model/             # Trained model files
├── Scripts/           # Python scripts
│   ├── Train_model.py
│   ├── Test_model.py
│   ├── DatasetCollector.py
│   └── requirement.txt
├── Logs/              # Training logs
├── createpath.py      # Utility script
└── README.md          # This file
```

## Performance Notes

- Real-time processing runs at ~30 FPS on standard hardware
- GPU recommended for faster training
- Recognition accuracy improves with more diverse training data

## Future Enhancements

- Support for more letters/numbers
- Multi-hand detection
- Word prediction using language models
- Mobile app deployment
- Model quantization for edge devices

## License

This project is open source and available for educational purposes.

## Contact

For issues, suggestions, or contributions, please open an issue on GitHub.
