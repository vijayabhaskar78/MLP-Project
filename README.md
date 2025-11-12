## System Threat Forecaster 🛡️
### Can You Forewarn a System Before It's Compromised?

[![AI Code Review](https://img.shields.io/badge/AI-Code%20Review%20Enabled-brightgreen)](https://github.com/vijayabhaskar78/MLP-Project)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### Overview
Welcome to the System Threat Forecaster, a Machine Learning Project (MLP T12025) from my BS Degree at IIT Madras. This project tackles the critical challenge of predicting whether a system will succumb to malware infections. Using telemetry data from antivirus software, I developed a model to forecast infection probabilities based on system properties. It's an exciting fusion of data science and cybersecurity—ideal for anyone eager to safeguard digital landscapes!

**🤖 AI-Powered Development:** This project features automated code reviews using Groq's LLaMA 3.3 model, ensuring high code quality and catching potential issues in pull requests automatically.

### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **Jupyter Notebook** or **JupyterLab**

### Installation
Follow these steps to set up the project locally:

```bash
# Clone the repository
git clone https://github.com/vijayabhaskar78/MLP-Project.git
cd MLP-Project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install pandas numpy scikit-learn catboost seaborn matplotlib jupyter

# Launch Jupyter Notebook
jupyter notebook
```

### The Challenge
**Goal:** Predict the likelihood of malware infection (0 or 1) using system properties.

**Dataset:** Real-world telemetry data from antivirus threat reports, rich with numerical and categorical features.

**Evaluation:** Measured by accuracy_score()—every prediction matters!

**Deadline:** Kicked off on Dec 26, 2024; set to conclude by Mar 19, 2025.

### My Journey: Hardships & Triumphs
Real-world data is chaotic, and this project was no exception. Here’s how I transformed obstacles into breakthroughs:

- **Missing Values:** Some columns had over 40% missing data! I dropped those and filled the rest—medians for numerical features, most frequent values for categories.
- **Outliers:** Numerical features showed extreme values. I applied the IQR method to trim them, boosting model robustness.
- **Complex Features:** Date and version strings were tricky. I crafted features like *DaysBetweenASandOS* and split version numbers into meaningful components.

### Approach: From Chaos to Clarity
#### Data Loading & Exploration:
- Loaded `train.csv` and `test.csv` using Pandas.
- Used Seaborn and Matplotlib to visualize missing values and outliers—bar plots and boxplots steered my preprocessing.

#### Data Preprocessing:
- Eliminated columns with >40% missing data.
- Filled gaps: categorical with mode, numerical with median.
- Trimmed outliers with IQR for cleaner data.

#### Feature Engineering:
- Converted dates into *DaysBetweenASandOS*.
- Split version columns (e.g., *EngineVersion*) into numerical sub-features for deeper insights.

#### Exploratory Data Analysis (EDA):
- Histograms showed distributions; bar plots revealed categorical trends.
- Correlation heatmaps exposed feature relationships.

#### Model Selection & Training:
- Started with a `DummyClassifier` to set a baseline.
- Moved to `LogisticRegression`, noting early improvements.
- Experimented with `RandomForestClassifier` using a pipeline for scaling and encoding, seeing better results.
- Settled on `CatBoostClassifier` for its native categorical data handling and top-tier performance.

#### Hyperparameter Tuning:
- Optimized `RandomForestClassifier` with `RandomizedSearchCV`.
- Fine-tuned `CatBoostClassifier` with `iterations=1000`, `learning_rate=0.05`, and early stopping to avoid overfitting.

#### Evaluation & Submission:
- Divided data into 80% training and 20% validation.
- Assessed performance with accuracy and AUC on the validation set.
- Produced `submission.csv` with id and target for the test set.

### Results
**Best Model:** `CatBoostClassifier`

**Validation AUC:** 0.6823

**Training Highlights:** The CatBoost model, trained with early stopping, hit its peak AUC of 0.6823 at iteration 662 out of 1000. It outperformed the `DummyClassifier`, `LogisticRegression`, and `RandomForestClassifier`, showing consistent AUC gains throughout training.

**Key Metrics:**
- Training Accuracy: 68.5%
- Validation Accuracy: 68.2%
- F1 Score: 0.67
- Precision: 0.69
- Recall: 0.65

### How to Run It
#### Clone the Repo:
```bash
git clone [your-repo-link]
cd system-threat-forecaster
```

#### Run the Notebook:
- Launch the provided Jupyter Notebook.
- Run all cells to preprocess, train, and predict.

#### Check Output:
- Locate `submission.csv` in the working directory.

### What's Next?
- **Feature Boost:** Explore advanced feature engineering, like interaction terms and polynomial features.
- **Model Ensemble:** Blend CatBoost with XGBoost or LightGBM for a performance lift.
- **Explainability:** Use SHAP to analyze feature importance and uncover malware predictors.
- **Cross-Validation:** Implement k-fold cross-validation for more robust model evaluation.
- **Hyperparameter Optimization:** Try Optuna or Grid Search for better parameter tuning.
- **Deployment:** Build a REST API using Flask or FastAPI for real-time predictions.

### Why This Matters
This project is more than code—it's a stride toward secure systems. I've sharpened my skills in data wrangling, feature engineering, and model optimization while addressing a pressing real-world issue. Recruiters, heads up: I'm primed to bring this analytical prowess to your team!

### Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

