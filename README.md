# Credit Risk Analysis with ML Models

This project focuses on analyzing credit risk using machine learning models. We use the UCI Credit Approval dataset to predict credit approval based on applicant data. Models include Random Forest and XGBoost (via Gradient Boosting).

## Project Structure
```
credit-risk-analysis/
│
├── data/
│   ├── crx.data
│   ├── crx.names
│
├── crx_processed.csv               # Cleaned and preprocessed dataset
│
├── scripts/
│   ├── preprocess.py               # Data cleaning and transformation
│   ├── train_model.py              # Model training and evaluation
│   ├── evaluate.py                 # Evaluation report generation
│
├── models/
│   ├── random_forest.pkl           # Trained Random Forest model
│   ├── xgboost.pkl                 # Trained XGBoost (GradientBoosting) model
│
├── output/
│   ├── evaluation_report.txt       # Final performance metrics of models
│
├── README.md
```

## Dataset
- **Source:** UCI Credit Approval Dataset
- **File:** `crx.data`
- **Attributes:** Mixed numerical and categorical features with missing values
- **Target Variable:** Class label (`+` approved / `-` rejected)

## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-analysis.git
cd credit-risk-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
1. Preprocess the data:
```bash
python scripts/preprocess.py
```

2. Train models:
```bash
python scripts/train_model.py
```

3. Generate evaluation report:
```bash
python scripts/evaluate.py
```

## Output
- Trained models are saved in the `models/` directory
- Evaluation results (accuracy, F1-score) are saved in `output/evaluation_report.txt`

## Author
Naveen Shinde
