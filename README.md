## 🌱 Plant Disease Detector

A **Convolutional Neural Network (CNN)** based project for detecting plant diseases from leaf images. This model uses **TensorFlow** and **Keras** to classify plant images into three categories with high accuracy.

---

### 🔍 **Overview**

This project builds a **deep learning CNN model** that classifies plant leaves into:

* **Corn – Common Rust**
* **Potato – Early Blight**
* **Tomato – Bacterial Spot**

The workflow includes:

* Data loading and preprocessing
* Image visualization
* CNN architecture design
* Model training & validation
* Prediction and evaluation
* Saving model and weights

---

## 📂 Dataset Structure

```
Plant_images_pianalytix/
 ├── Corn_Common_rust/
 ├── Potato-Early_blight/
 └── Tomato-Bacterial_spot/
```

---

## 🧰 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy & Pandas**
* **OpenCV**
* **Matplotlib**
* **Scikit‑learn**
* **Google Colab**

---

## 🛠️ Data Preprocessing

### **✔ 1. Mount Google Drive**

Used to load dataset in Colab.

### **✔ 2. Visualize Sample Images**

Matplotlib is used to display random images to verify dataset integrity.

### **✔ 3. Convert Images to Arrays**

All images are:

* Read with OpenCV
* Resized to **256×256**
* Converted into NumPy arrays

### **✔ 4. Encode Labels**

```
Corn_Common_rust → 0
Potato-Early_blight → 1
Tomato-Bacterial_spot → 2
```

Labels are one‑hot encoded.

### **✔ 5. Split Dataset**

* **80% train**
* **20% test**
* Extra **validation split** from the training set

---

## 🧠 CNN Model Architecture

The model contains:

* **Conv2D + ReLU Activation**
* **MaxPooling2D**
* **Flatten Layer**
* **Dense Layer (64 units)**
* **Softmax Output Layer (3 classes)**

### **Model Compilation**

* Loss: `categorical_crossentropy`
* Optimizer: **Adam(0.0001)**
* Metric: **accuracy**

---

## 🏋️ Model Training

* **Epochs:** 50
* **Batch Size:** 128
* Training accuracy and validation accuracy are plotted.

---

## 📊 Performance Visualization

Training curves show how accuracy improves over time.

---

## 💾 Saving the Model

Saved in multiple formats:

* `plant_disease.h5` – full model
* `plant_disease.json` – model architecture
* `plant_model_weights.weights.h5` – weights only

---

## Prediction Example

Model predicts class labels from test images using:

```
y_pred = model.predict(x_test)
```

Example result:

```
Originally : Potato‑Early_blight
Predicted : Potato‑Early_blight
```

---

## 🏁 Conclusion

The project successfully builds a **CNN-based plant disease classifier** with strong accuracy. It demonstrates:

* Proper dataset handling
* Clean preprocessing pipeline
* Well‑structured CNN architecture
* Accurate predictions on unseen data

---

## Future Enhancements

* Add more plant disease classes
* Use **data augmentation**
* Apply **transfer learning** (e.g., MobileNet, ResNet)
* Deploy as mobile/web application
