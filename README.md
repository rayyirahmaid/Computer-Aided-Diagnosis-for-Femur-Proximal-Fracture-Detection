# Computer-Aided-Diagnosis-for-Femur-Proximal-Fracture-Detection
##A deep learning YOLO-based system for detecting femur proximal fracture

This repository contains a desktop application designed for the detection of proximal femur fractures from X-ray images. The application leverages a deep learning model, YOLO11-N, which was trained on an X-ray AP pelvis open source dataset.The application provides a user-friendly graphical interface built with tkinter, allowing for real-time fracture detection, image manipulation, and analysis of model predictions. It is a comprehensive tool for both clinical use and training/evaluation purposes.

## Key Features: 
- Deep Learning-based Detection: Utilizes a powerful YOLO11-N model to accurately identify proximal femur fractures.
- Intuitive GUI: A desktop application built with tkinter that allows users to easily load X-ray images, run the detection model, and view results.
- Multiple Modes:
    - Detection Mode: Provides a streamlined workflow for immediate fracture detection and result visualization.
    - Training Mode: Enables users to annotate images and evaluate the model's performance against their annotations, providing metrics like IoU (Intersection over Union) and bounding box count error.
    - Guidance Mode: Displays model predictions as a guide for users, allowing them to create their own annotations with the help of the model's output.
- Image Manipulation Tools: Includes features for applying various view modes (Grayscale, Inverted, CLAHE) and transformations (brightness, contrast, flipping, rotation) to enhance image analysis.
- Detailed Evaluation: Offers a comprehensive evaluation system to compare user annotations with model predictions, generating evaluation metrics and saving the results for documentation.

## Repository Structure

The application's source code is organized into a modular structure to facilitate maintenance and scalability.
- main.py: The entry point of the application. Initializes the main GUI window and runs the application.
- model_handler.py: Manages the YOLO11-N model, including loading the model, running predictions, and applying Non-Maximum Suppression (NMS) to filter redundant bounding boxes.
- gui_components.py: Contains the FractureDetectionApp class, which defines the entire GUI layout, event handling, and logic for all application modes and features.
- image_processor.py: Provides a set of static methods for loading, processing, and manipulating images for display and analysis.
- evaluation.py: Includes classes and methods for evaluating model predictions against user annotations, calculating metrics such as IoU and bounding box count error.
- utils.py: A collection of utility classes, including a ConfigManager for handling application settings and a Logger for tracking application activities.
- models/: This directory is intended to store the trained model file (best.pt), which is essential for the application's functionality.

## Requirements

To run this application, you will need to have Python and the required libraries installed. You can install the dependencies by running the following command:

>`pip install -r requirements.txt`

## Usage

1. Clone this repository:
> `git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)`

> `cd your-repo-name`

2. Place your trained YOLO model file `best.pt` inside the `femur_fracture_app/models/` directory.
3. Run the application:
`python main.py`

Feel free to explore the code, contribute, or adapt it for your own projects.
