
# Lung Cancer Detection on IQ-OTH/NCCD Dataset 🫁

This project is an end-to-end pipeline for lung cancer detection using the **IQ-OTH/NCCD dataset**. The pipeline includes multiple deep learning models (CNN, RNN, ANN), extensive explainability (Grad-CAM, SHAP, LIME), and detailed performance evaluation metrics. It is designed for efficient training and testing with optimizations for integrated graphics (Intel Iris Xe).

## 🚀 Features

- ✅ Full pre-processing and loading pipeline for IQ-OTH/NCCD dataset
- ✅ Model training and evaluation for:
  - Convolutional Neural Network (CNN)
  - Recurrent Neural Network (RNN)
  - Artificial Neural Network (ANN)
- ✅ Model explainability:
  - Grad-CAM visualization
  - SHAP explainability
  - LIME interpretability
- ✅ Performance evaluation:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - Training/Validation accuracy and loss curves
  - Final model comparison bar charts
- ✅ Optimized for Intel Iris Xe integrated graphics
- ✅ Inline display of explainability visuals and performance plots
- ✅ Model saving and loading support

## 📂 Dataset

- **Name:** IQ-OTH/NCCD Lung Cancer Dataset
- **Structure:**

  ```
  /train
    /class0
    /class1
  /test cases
    *.png
  ```

- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sergiuoprescu/iqothnccd-lung-cancer-dataset).

## 🛠️ Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras shap lime opencv-python tqdm
```

For Intel GPU acceleration:

```bash
pip install intel-opencl-rt
```

> 💡 Optional: For better visualizations in Jupyter, ensure you are using `matplotlib-inline`.

## 📒 How to Run

1. Clone this repository and navigate to the project directory.
2. Install the required packages.
3. Open the notebook:

   ```bash
   jupyter notebook iq-oth-nccd-lung-cancer-detection.ipynb
   ```

4. Follow through the notebook sections:
   - Dataset loading and preprocessing
   - Model training (CNN, RNN, ANN)
   - Model evaluation
   - Explainability visualizations
   - Model comparisons

## 📊 Results

The notebook provides:
- Training and validation accuracy/loss plots
- Model performance comparisons (bar charts)
- Inline Grad-CAM heatmaps for CNN
- SHAP summary plots
- LIME local explanations

All models are trained and evaluated separately for better interpretability and comparison.

## 🧩 Explainability

- **Grad-CAM:** Visualizes important regions for CNN model predictions.
- **SHAP:** Global and local explanations for model outputs.
- **LIME:** Local interpretability for individual predictions.

## 💾 Saving & Loading Models

Trained models are saved in the working directory and can be reloaded for inference or further analysis.

## 📈 Future Improvements

- ✅ Data augmentation
- ✅ Hyperparameter tuning
- ✅ Cross-validation support
- ✅ Integration with web app for real-time predictions

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## 📄 License

This project is open-source and available under the MIT License.
