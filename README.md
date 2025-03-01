# Human Activity Recognition using Smartphone Sensors

This repository contains implementations of both deep learning and traditional machine learning approaches for human activity recognition (HAR) using the UCI HAR Dataset. The goal is to classify human activities (walking, sitting, etc.) based on smartphone sensor data.

## Dataset

The UCI HAR Dataset consists of accelerometer and gyroscope readings from smartphones, capturing data as subjects performed six activities:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

### Data Structure
- **Raw signals**: 9 channels (3 for total acceleration, 3 for body acceleration, 3 for angular velocity)
- **Window size**: 128 timesteps per window (2.56 seconds at 50Hz)
- **Features**: Both raw signals and a set of 561 pre-computed features provided by the dataset authors

## Approaches

This project explores two complementary approaches to HAR:

### Deep Learning Approach

Two neural network architectures were implemented to learn directly from the raw sensor signals:

#### LSTM Model
- Two-layer LSTM network with hidden dimension of 64
- Captures temporal dependencies in the sensor data
- Final hidden state passed to a fully connected layer for classification

#### 1D CNN Model
- Two 1D convolutional layers with ReLU activations and max pooling
- Processes signals across the time dimension
- Flattened feature maps fed into a fully connected layer for prediction

Both models were trained using:
- CrossEntropyLoss function
- Adam optimizer with learning rate of 0.001
- 200 training epochs

### Machine Learning Approach

This approach relies on feature extraction followed by traditional classification algorithms:

#### Feature Extraction
- **TSFEL-extracted features**: Automatic computation of time and frequency domain features using the Time Series Feature Extraction Library
- **Provided features**: 561 pre-computed features from the original dataset

#### Classifiers
Three machine learning algorithms were evaluated:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

Each classifier was implemented in a pipeline with StandardScaler for feature normalization.

## Results

### Deep Learning Performance
- **LSTM**: ~91.45% accuracy
- **1D CNN**: ~92.13% accuracy

Common misclassifications occurred between similar activities like SITTING vs. STANDING or STANDING vs. LAYING, which involve low motion patterns.

### Machine Learning Performance
All three classifiers achieved strong results:
- Best performance from SVM and Logistic Regression (>95% accuracy)
- TSFEL-extracted features slightly outperformed the provided features for SVM and Logistic Regression
- Random Forest performed well but with slightly lower accuracy than the other models

### Key Observations
1. Both deep learning and traditional machine learning approaches are effective for HAR, achieving >90% accuracy
2. The 1D CNN slightly outperformed LSTM in the deep learning comparison
3. Automated feature extraction (TSFEL) can yield results comparable to or better than carefully engineered features
4. Activities with similar body postures or transitions are more challenging to distinguish for all models


## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- TSFEL
- numpy
- pandas
- matplotlib

## Future Work

- Explore hybrid models combining deep learning and traditional features
- Implement more complex architectures (e.g., attention mechanisms, transformers)
- Test transfer learning approaches
- Evaluate performance on real-time data streams

## Citation

If you use this code or the UCI HAR Dataset in your research, please cite:

```
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. 
A Public Domain Dataset for Human Activity Recognition Using Smartphones. 
21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. 
Bruges, Belgium 24-26 April 2013.
```
