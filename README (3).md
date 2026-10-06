# MLEnd Hums and Whistles II: Song Classification 🎵

This repository contains my mini-project for ECS7020P Principles of Machine Learning, where I predict the title of a song from a 10-second hummed or whistled recording.


## About the Project ❓

The task is a multi-class classification problem: given a 10-second audio recording of someone humming or whistling a song, predict which song it is. The project uses the 400-sample subset of the **MLEnd Hums and Whistles II Dataset**, which contains 8 songs with 50 recordings each.

Raw audio is converted into fixed-length numerical features (MFCC and Chroma), and three machine learning pipelines are trained and compared. The aim of the project is to show a sound methodology and honest evaluation, not to reach a high score on a very difficult problem.

For practical implementation, Python and its libraries for audio processing, scientific computing, visualisation and machine learning are used.


## Environment 👩🏻‍💻

<p align="center">
  <img src="https://img.shields.io/badge/jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" alt="GitHub"/>
</p>


## Stack 🛠️

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/librosa-FF6F00?style=flat" alt="librosa"/>
  <img src="https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=flat" alt="matplotlib"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
</p>


## Methodology 🧪

| Stage | What I did |
|---|---|
| **Input** | 10-second audio recordings (400 samples, 8 songs, 50 recordings per song) |
| **Pre-processing** | Resampled to 22,050 Hz, padded or trimmed to exactly 10 seconds, normalised amplitude |
| **Feature extraction** | **MFCC** (spectral shape, with delta and delta-delta) and **Chroma** (pitch content) |
| **Fixed-length vectors** | Summarised each feature over time using its mean and standard deviation |
| **Data split** | Stratified 70/30 split: 280 training and 120 validation samples |
| **Models** | Logistic Regression (baseline) and Support Vector Machine (SVM) |
| **Evaluation** | Accuracy, balanced accuracy, classification report and confusion matrices |

### Pipelines
- **Pipeline A:** MFCC features + Logistic Regression
- **Pipeline B:** MFCC features + SVM
- **Pipeline C:** Chroma features + SVM


## Results 📊

| Pipeline | Features | Classifier | Validation Accuracy |
|---|---|---|---|
| A | MFCC | Logistic Regression | 0.2083 |
| B | MFCC | SVM | 0.2167 |
| **C** | **Chroma** | **SVM** | **0.3250** |

With 8 balanced classes, random guessing would score about 12.5%, so all three pipelines beat chance. Pipeline C (Chroma + SVM) performed best, and its confusion matrix showed fewer misclassifications than A and B.


## Key Takeaways 💡

- Chroma features worked best, because they capture pitch, which is what humming and whistling carry.
- SVM beat Logistic Regression on MFCC features, though only slightly.
- All three pipelines scored above the 12.5% chance level, even though the task is very hard.
- Several recordings may come from the same person, so the validation set is not fully independent.
- A larger dataset (the 800-sample set) and richer features, such as Mel spectrograms or combined features, could improve results.


## Repository Structure 🌲

```
.
├── .gitattributes
├── Miniproject.ipynb
└── README.md
```


## Reflection 🪞

This mini-project gave me hands-on experience with a full machine learning workflow on audio data, from loading and standardising recordings to extracting features, building pipelines, and evaluating them with confusion matrices. The results showed me that a low accuracy can still be a meaningful result when the methodology is sound and the limitations are understood, such as the small sample size and the possibility that one person recorded several samples.

Looking ahead, I want to try the larger 800-sample dataset, add Mel spectrogram features, and explore ensemble methods and group-aware validation to get more reliable estimates.
