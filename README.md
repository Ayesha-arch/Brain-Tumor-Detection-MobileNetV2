# AI-Based Brain Tumor Detection and Analysis

![GitHub last commit](https://img.shields.io/github/last-commit/YourGitHubUsername/Brain-Tumor-Detection-MobileNetV2?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/YourGitHubUsername/Brain-Tumor-Detection-MobileNetV2?style=flat-square)
![License](https://img.shields.io/github/license/YourGitHubUsername/Brain-Tumor-Detection-MobileNetV2?style=flat-square)

## Project Overview

This repository presents an end-to-end deep learning solution for the detection, classification, segmentation, and severity assessment of brain tumors from MRI scans. Leveraging the efficient MobileNetV2 architecture (TensorFlow/Keras) with transfer learning, the system provides accurate predictions across four classes: glioma, meningioma, pituitary, and no-tumor.

This project aims to demonstrate an effective, interpretable, and deployable AI tool to assist in the preliminary analysis of brain MRI images. It is ideal for medical imaging research and development, offering a comprehensive pipeline from data loading and model training to an interactive user interface and report generation.

## Key Features

*   **Brain Tumor Classification:** Utilizes a fine-tuned MobileNetV2 model for high-accuracy classification of tumor types.
*   **Image Enhancement & Segmentation:** Incorporates robust image preprocessing and a custom segmentation pipeline to accurately delineate tumor regions.
*   **Severity Detection:** Classifies tumor severity based on segmented tumor size (in pixels).
*   **AI-Powered Clinical Narratives:** Integrates the Groq API (Llama 3.3) to generate context-aware, professional clinical suggestions and interpretations, acting as a "Senior Neuroradiologist."
*   **Interactive Web UI:** Features a user-friendly Gradio interface for real-time MRI image analysis, patient data input, and instant results visualization.
*   **PDF Report Generation:** Automatically generates comprehensive, downloadable PDF medical reports including patient information, original MRI, tumor overlay, tumor mask, AI analysis, and clinical suggestions.
*   **Robust Data Handling:** Implements custom data generators for memory-efficient training and on-the-fly image augmentation (Gaussian Blur, CLAHE).

## Technologies Used

*   **Programming Language:** Python 3.x
*   **Deep Learning Framework:** TensorFlow 2.x / Keras
*   **Computer Vision:** OpenCV (`cv2`)
*   **Data Manipulation:** Pandas, NumPy
*   **Model Architectures:** MobileNetV2 (pre-trained on ImageNet)
*   **Machine Learning Utilities:** scikit-learn (for `LabelBinarizer`, `train_test_split`)
*   **Interactive UI:** Gradio
*   **PDF Generation:** FPDF2
*   **Large Language Model Integration:** Groq API (`llama-3.3-70b-versatile`)
*   **Visualization:** Matplotlib
*   **Utilities:** `tqdm`, `zipfile`, `os`, `datetime`, `pytz`

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourGitHubUsername/Brain-Tumor-Detection-MobileNetV2.git
    cd Brain-Tumor-Detection-MobileNetV2
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Google Colab Setup (for full functionality including Gradio and Groq):**
    *   Upload `MRI Dataset.zip` to your Colab environment or place it in the project root.
    *   Ensure you have a `GROQ_API_KEY` set up in Colab's Secrets Manager (`🔑` icon on the left panel).

## Usage

1.  **Prepare your dataset:** Place your `MRI Dataset.zip` (containing `train` and `test` folders with `glioma`, `meningioma`, `pituitary`, `notumor` subdirectories) in the appropriate location as specified in the notebook (e.g., upload via `files.upload()` in Colab or ensure it's in `/content/dataset` after unzipping).

2.  **Run the Colab Notebook:** Execute the cells sequentially in the provided Google Colab notebook (`your_notebook_name.ipynb`). The notebook will guide you through:
    *   Installing dependencies.
    *   Loading and preprocessing MRI images.
    *   Training the MobileNetV2 model.
    *   Evaluating the model.
    *   Launching the Gradio interactive UI.

3.  **Using the Gradio Interface:**
    Once the Gradio interface is launched (usually at the end of the notebook), you can:
    *   Upload an MRI image.
    *   Enter patient details (Name, Age, Gender).
    *   Click "Submit" to get:
        *   Original MRI, Tumor Overlay, Tumor Mask visualizations.
        *   Predicted Tumor Type, Confidence, Tumor Size, and Severity.
        *   A downloadable PDF medical report with AI-generated clinical suggestions.

## Dataset

The project utilizes a custom MRI dataset (`MRI Dataset.zip`) consisting of approximately 5736 brain MRI images, categorized into `glioma`, `meningioma`, `pituitary`, and `notumor` types. Images are preprocessed by resizing to 256x256 pixels, normalizing to `[0,1]`, and enhanced on-the-fly using Gaussian blur and CLAHE. The dataset is split into 80% for training and 20% for testing, with stratified sampling.

## Model Performance

The MobileNetV2 model, trained with transfer learning and optimized with Early Stopping and ReduceLROnPlateau callbacks, achieved the following performance on the test set:

*   **Test Accuracy:** ~94.76%
*   **Test Loss:** ~0.1715

These results demonstrate the model's strong capability in classifying brain tumors, providing a solid foundation for further clinical validation and deployment.

## Project Structure (Colab Notebook Flow)

The Colab notebook is structured logically:

1.  **Setup:** Installs required libraries and imports necessary modules.
2.  **Dataset Handling:** Uploads, unzips, loads, and preprocesses the MRI dataset.
3.  **Image Enhancement:** Defines and initializes custom batch-safe generators for image enhancement.
4.  **Model Definition:** Sets up the MobileNetV2 (and ResNet50 for comparison, though MobileNetV2 is trained) architecture for classification.
5.  **Training:** Trains the MobileNetV2 model using the custom generators and callbacks.
6.  **Evaluation:** Assesses the trained model's performance on the test set.
7.  **Prediction & Segmentation:** Implements functions for individual image prediction, tumor segmentation, and severity detection.
8.  **Interactive UI & Reporting:** Integrates Gradio for an interactive demo and FPDF2 for generating comprehensive medical reports, including AI-powered clinical insights.
9.  **Model Saving:** Saves the trained MobileNetV2 model for future use.
10. **Requirements.txt Generation:** Creates a file listing all project dependencies.

## Future Work & Improvements

*   **Dataset Expansion:** Integrate larger, more diverse, and publicly available brain tumor datasets (e.g., from TCGA, BraTS challenges) for improved generalization.
*   **Model Comparison:** Empirically train and evaluate the ResNet50 model (already defined in the notebook) and potentially other state-of-the-art CNN architectures.
*   **Advanced Segmentation:** Explore more sophisticated segmentation models (e.g., U-Net, Attention U-Net) for more precise tumor boundary detection.
*   **Explainable AI (XAI):** Integrate techniques like Grad-CAM or SHAP to provide visual explanations for model predictions, enhancing trust and interpretability.
*   **Quantification of Tumor Growth/Change:** Implement methods to compare successive MRI scans to track tumor evolution over time.
*   **Clinical Validation:** Conduct rigorous clinical validation with medical professionals to assess real-world applicability and impact.
*   **Deployment Optimization:** Optimize the model for edge deployment or cloud inference services for faster processing.

## Contributing

Contributions are welcome! Please feel free to fork the repository, open issues, or submit pull requests to improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact [Your Name/Email/LinkedIn Profile].
