# VisionWrite – Real-Time Hand Gesture & Air-Writing Recognition

VisionWrite is a computer vision and machine learning project that explores hand-based interaction through real-time webcam input. It combines shape detection, digit recognition, and hand gesture recognition to create an interactive system.

The project is divided into three phases, each focusing on a different recognition task.

## Project Phases

### Phase 1: Geometric Shape Detection
- Detects and identifies geometric shapes using computer vision.
- Processes visual input to recognize different shapes.

### Phase 2: Air-Writing Digit Recognition (0–9)
- Recognizes handwritten digits from 0 to 9.
- Uses a trained machine learning/deep learning model for digit classification.
- Explores handwritten character recognition.

### Phase 3: Sign Language Alphabet Recognition (A–Z)
- Focuses on recognizing hand signs representing alphabet letters.
- Uses hand landmark detection and collected gesture data.
- Uses a trained model to classify alphabet gestures.

**Note:** This phase focuses on individual alphabet signs (fingerspelling), not full sentence-level sign language translation.

## Technologies Used

- Python
- OpenCV – Computer vision and webcam processing
- MediaPipe – Hand landmark detection
- TensorFlow / Keras – Deep learning and model training
- NumPy – Numerical operations
- Machine Learning and Convolutional Neural Networks (CNN)

## Project Structure

```text
VisionWrite/
│
├── Phase1_shapedetection
├── Phase2_0-9_writing
├── Phase3_alphabet_recog
├── Collect_signs
├── Train_letters
├── Letters_mod.keras
├── Sign_model.keras
├── Sign_data.npy
├── Sign_labels.npy
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/atufasuhail098-wq/VisionWrite.git
```

2. Open the project folder:

```bash
cd VisionWrite
```

3. Install the required libraries:

```bash
pip install opencv-python numpy tensorflow mediapipe
```

4. Run the relevant Python scripts for each project phase.

## Learning Objectives

- Understand real-time computer vision applications.
- Explore hand landmark detection using MediaPipe.
- Implement machine learning and deep learning models.
- Understand image and gesture classification.
- Build interactive applications using webcam input.

## Future Improvements

- Improve recognition accuracy.
- Make gesture detection smoother and more reliable.
- Expand gesture vocabulary.
- Improve real-time performance.
- Develop a more interactive user interface.

## Author

**Atoofa Suhail**

MSc Data Science Student

---

*VisionWrite is an academic and learning project developed to explore computer vision, machine learning, and human-computer interaction.*
