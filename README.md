# Brazilian E-commerce Review Classification System

A comprehensive machine learning pipeline for classifying Brazilian e-commerce review sentiment with structured reason extraction.

---

## Project Overview

This project implements an end-to-end machine learning solution for the **Brazilian E-commerce Review Classification** challenge. It processes review data from the Olist dataset to:

- Predict sentiment (positive/negative)  
- Extract structured reasons behind customer opinions  

**Challenge Requirements**
- **Primary Goal:** Classify product reviews as positive or negative  
- **Data Source:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
- **Key Challenge:** Create target variable from review scores and text (no direct sentiment labels)  
- **Bonus:** Extract structured reasons (delivery, quality, service, price)  

---

## Features

### Core Functionality
- Multi-model Classification: Logistic Regression, Random Forest, Gradient Boosting with hyperparameter tuning  
- Portuguese Text Processing: NLP pipeline tailored for Brazilian Portuguese reviews  
- Target Variable Creation: Conversion of 1–5 star ratings into binary sentiment  
- Feature Engineering: 50+ engineered features across temporal, geographic, behavioral, and text data  

### Bonus Features
- Sentiment Reason Extraction (delivery, quality, service, price)  
- Data Quality Assessment with automated validation  
- Interactive Visualizations  
- Modular Architecture for production-ready code  

---

## Project Structure

## 📂 Project Structure

```plaintext
brazilian_ecommerce_classifier/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── quality.py
│   │   ├── validator.py
│   │   ├── review_analyzer.py
│   │   └── merger.py
│   ├── preprocessing/
│   │   ├── target_creator.py
│   │   ├── text_processor.py
│   │   └── feature_engineer.py
│   └── modeling/
│       ├── classifier.py
│       └── reason_extractor.py
├── data/              # Raw CSV files (not included)
├── main.py            # Main pipeline orchestration
└── requirements.txt   # Python dependencies


---

## ⚙️ Installation

### Prerequisites
- Python 3.8+  
- 4GB+ RAM (for full dataset processing)  
- Kaggle account + Olist dataset  

### Setup
```bash
# Clone repository
git clone <repository-url>
cd brazilian_ecommerce_classifier

# Install dependencies
pip install -r requirements.txt
```

### Dataset
- Download dataset: Brazilian E-Commerce Public Dataset by Olist  
- Place all CSV files into the `data/` folder  

### Run Pipeline
```bash
python main.py
```

---

## Dependencies

- pandas >= 1.3.0  
- numpy >= 1.21.0  
- scikit-learn >= 1.0.0  
- matplotlib >= 3.5.0  
- seaborn >= 0.11.0  

**Optional (NLP):**  
- spacy >= 3.4.0  
- textblob >= 0.17.0  

---

## Usage

### Basic Example
```python
from src.data import load_olist_data, assess_data_quality_local
from src.preprocessing import create_classification_target, preprocess_portuguese_text
from src.modeling import build_classification_models

datasets = load_olist_data('data/')
quality_report = assess_data_quality_local(datasets)
df_with_target = create_classification_target(master_df)
model_results = build_classification_models(df_engineered)
```

### Example (Bonus Implementation)
```python
from src.modeling import extract_sentiment_reasons

reason_results = extract_sentiment_reasons(df_engineered, model_results)
print(f"Reason coverage: {reason_results['coverage_percentage']:.1f}%")
```

---

## Methodology

### Data Pipeline
- Load & validate 9 CSVs  
- Assess data quality (missing, duplicates, outliers)  
- Merge orders, items, customers, sellers, payments, reviews  

### Target Creation
- 1–2 stars = Negative  
- 4–5 stars = Positive  
- 3 stars = Excluded  

### Text Preprocessing
- Handles accents, contractions, stopwords  
- Extracts sentiment keywords & text features  

### Feature Engineering
- Temporal, behavioral, geographic, and price-related features  

### Model Training
- Logistic Regression, Random Forest, Gradient Boosting  
- Grid search + cross-validation  
- Macro F1-score evaluation  

### Reason Extraction (Bonus Implementation)
- Delivery, quality, service, price, product description  

---

## Results

- **Best Model:** (output shown after running pipeline)  
- Accuracy: ~85–90%  
- F1-Score: Balanced  

**Insights**  
- 41% of reviews contain text  
- Delivery time is the strongest predictor of sentiment  
- Geographic region influences satisfaction  
- Price sensitivity varies by category  

**Bonus**  
- Reason coverage: 60–80%  
- Delivery & product quality most predictive  

---

## Technical Highlights

- Modular code structure  
- Robust error handling & logging  
- Efficient data processing & parallel training  

---

## Business Value

- Identifies key drivers of customer satisfaction  
- Provides actionable insights for delivery, product quality, and service improvements  
- Scalable framework for other datasets/platforms  

---

## Limitations

- Optimized for Brazilian Portuguese only  
- Model relies on text availability (~41% reviews)  
- Computationally intensive on full dataset  

---

## Future Enhancements

- Transformer models (BERT)  
- Real-time streaming support  
- Multilingual expansion  
- Deep learning integration  
- REST API for deployment  

---


