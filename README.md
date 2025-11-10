🎬 Sentiment Analysis on Movie Reviews

This project performs sentiment classification on IMDB movie reviews using PySpark and Transformer-based embeddings (Sentence-BERT).
It combines the scalability of Big Data (PySpark) with the accuracy of modern NLP models to predict whether a review expresses positive or negative sentiment.

🚀 Project Overview

The goal of this project is to build an end-to-end sentiment analysis pipeline that can efficiently handle large-scale textual data.
It integrates:

PySpark for distributed data processing

Sentence-BERT for semantic embeddings

Logistic Regression (Spark ML) for classification

Matplotlib/Seaborn for visualization and performance analysis

🧠 Features

✅ Preprocessing of raw IMDB reviews using Spark DataFrames
✅ Text embedding generation via Sentence-BERT (Transformer Model)
✅ Scalable model training with Spark ML Logistic Regression Pipeline
✅ Evaluation metrics including Accuracy, F1-Score, ROC Curve
✅ Visual analytics using Matplotlib and Seaborn

🧰 Tech Stack
Category	Tools & Libraries
Programming	Python, PySpark
NLP	Transformers, Sentence-BERT
ML Pipeline	Spark MLlib (VectorAssembler, LogisticRegression)
Visualization	Matplotlib, Seaborn
Environment	Google Colab / Jupyter Notebook
📂 Project Structure
📁 Sentiment-Analysis-on-Movie-Reviews
│
├── Main_Copy_of_TWS_Mini_Project.ipynb     # Main notebook
├── IMDB Dataset.csv                        # Dataset (or provide Kaggle link)
├── README.md                               # Project documentation
└── requirements.txt                        # List of dependencies

📊 Workflow

Data Loading

Load IMDB dataset (50,000+ reviews).

Convert to PySpark DataFrame for parallel processing.

Preprocessing

Text cleaning, tokenization, and handling of null/missing values.

Embedding Generation

Use Sentence-BERT to convert text into semantic numerical vectors.

Model Training

Build a Logistic Regression model using Spark ML pipeline.

Split data into train/test sets and evaluate performance.

Evaluation & Visualization

Measure accuracy, F1-score, and ROC AUC.

Plot confusion matrix and distribution of sentiments.

📈 Results
Metric	Score
Accuracy	~88%
F1-Score	~0.87
ROC AUC	~0.90

(Scores may vary slightly depending on dataset sampling and hyperparameters.)

🧩 Key Insights

Sentence-BERT embeddings outperform traditional TF-IDF in capturing context and semantics.

Using PySpark enables efficient processing of large-scale text data.

Logistic Regression provides interpretable and consistent classification results.

🔧 Installation

Clone the repository

git clone https://github.com/omsagvekar/Sentiment-Analysis-on-Movie-Reviews
cd sentiment-analysis-pyspark


Install dependencies

pip install -r requirements.txt


Run the notebook

jupyter notebook Main_Copy_of_TWS_Mini_Project.ipynb

📚 Future Enhancements

Integrate Power BI or Streamlit Dashboard for interactive visualization.

Add real-time sentiment API using Flask or FastAPI.

Experiment with DistilBERT and RoBERTa for improved contextual understanding.

👨‍💻 Author

Om Sunil Sagvekar
📧 omsagvekar04@gmail.com

🔗 LinkedIn : www.linkedin.com/in/om-sagvekar
 | GitHub : https://github.com/omsagvekar