# German-Traffic-Sign-CNN-Private
A Structured Project Practicum where CNN models were trained to classify German Traffic Signs using the public GTSRB dataset.
# 🚦 Autonomous Driving Traffic Sign Recognition: Convolutional Neural Network (CNN) Pipeline

An end-to-end deep convolutional neural network (CNN) classification pipeline developed to automatically identify real-world road traffic signs from camera inputs. This system processes a localized subset of the German Traffic Sign Recognition Benchmark (GTSRB) dataset across safety-critical categories to enable high-accuracy predictive modeling for computer vision in advanced driver-assistance systems (ADAS).

> 🔒 **Academic Integrity & Compliance Notice:** In strict accordance with the **[HKUST Academic Honor Code](https://hkust.edu.hk)**, all raw executable source code files (`.py`, `.ipynb`) and computational model training assets are securely maintained in a **Private** repository to prevent unauthorized distribution and duplication. A comprehensive structural overview, engineering layout documentation, and model performance metrics are completely documented below. Codebase access verification can be explicitly granted to recruiters upon requested review.

> 📝 **Academic Program Context (Structured Practicum):** This project was developed as a heavily structured programming practicum within the **HKUST COMP 2211 (Exploring Artificial Intelligence)** curriculum. The baseline training workspace scaffolding, template image directory loaders, hidden test harnesses, and task guidelines were provided by course instructors and teaching assistants. All custom core engineering implementations—specifically the matrix-stratified training allocation mathematical partitions, custom layered neural architectural topologies, layer-wise pooling configurations, and regularized hyperparameter training limits—were executed independently.

---

## 🚀 Key Project Achievements
* **Peak Test Validation Accuracy:** Secured a top-tier classification accuracy of **99.27%** on verified cross-validation sets.
* **Loss Minimization Stability:** Optimized categorical cross-entropy loss convergence down to a minimal validation bound of **~0.027**.
* **Trainable Parameter Efficiency:** Configured an optimized network utilizing under **422,000 trainable parameters**, maximizing training speed and ensuring strict parameter safety boundaries.
* **Checkpoint Weight Recalls:** Integrated callback serialization routines to automatically save and restore the highest performing historical model weight vectors.

---

## 📊 Dataset Specifications
* **Core Source Material:** Utilized the open-source **German Traffic Sign Recognition Benchmark (GTSRB)**, a benchmark consisting of real-world multi-class road sign imagery captured under volatile ambient lighting conditions, distances, and camera angles.
* **Target Isolation Partitioning:** For the purposes of this specialized practicum, the dataset was isolated and downsampled into a robust subset containing exactly **5 distinct categorical traffic sign classes** encompassing **2,210 distinct samples** sized to uniform 30x30 spatial domains.

---

## 📊 Model Training Performance Metrics

The neural network was trained over 30 epochs using an immutable model checkpoint tracking routine to record optimization trends:

### 1. Data Splitting Stratification & Preprocessing (`train_test_split`)
* **Proportional Stratification Split:** Engineered a mathematical partition to split the image directory arrays into a **75% training matrix (1,650 samples)** and a **25% validation matrix (550 samples)**. Applied multi-class class stratification to safeguard uniform class distributions across both matrices and eliminate test leakage.
* **Pixel Value Normalization:** Applied element-wise scalar transformations to re-scale raw integer pixel coordinates from `[0, 255]` down to normalized float coefficients in the bound of `[0.0, 1.0]`, guaranteeing smooth vector gradients during backward propagation.
* **One-Hot Vectorization Encoding:** Transformed absolute integer index targets into 5-dimensional binary array maps to facilitate structural classification compliance with multi-output compilation cross-entropy objectives.

### 2. CNN Network Architecture Optimization (`build_model`)
* **Feature Extraction Core Layers:** Structured a `Sequential` network flow deploying 2D Convolution (`Conv2D`) blocks integrated with 3x3 sliding spatial feature extraction kernels utilizing rectified linear unit (`relu`) activations.
* **Dimensionality Suppression & Matrix Reduction:** Implemented max-pooling filters (`MaxPooling2D`) configured with 2x2 pooling matrices to cleanly compress spatial feature depths while tracking dominant pixel activations.
* **Overfitting Regularization Safe-Checks:** Inserted structural `Dropout` regularizers throughout deep hidden nodes—scaling from a 25% dropout rate up to a 50% rate on deep parameters—to effectively break weight co-dependencies and eliminate validation variance.
* **Dense Topography Output Head:** Compressed multi-spectral spatial hidden depths using a structural `Flatten` layer into a 128-neuron dense hidden network before routing to a final 5-neuron `Dense` Softmax layer mapping the conditional probability matrix of each road sign class.

---

## 🧰 Technologies & Toolkits Used
* **Programming Languages:** Python
* **Deep Learning Frameworks:** Keras, TensorFlow
* **Scientific Computing & Data Wrangling:** NumPy, Scikit-Learn 
* **Visualization Libraries:** Matplotlib
* **Development Environments:** Google Colab
