🎬 Sentiment Analysis on Movie Reviews

Sentiment Analysis on Movie Reviews is an end-to-end sentiment classification project built with PySpark and Sentence-BERT. The goal is to predict whether a movie review expresses a positive or negative sentiment, utilizing distributed processing with PySpark and semantic embeddings via Sentence-BERT.

🚀 Features

Preprocessing: Clean, tokenize, and handle missing values in movie reviews.

Sentence-BERT: Generate semantic embeddings for better text representation.

Logistic Regression: Use Spark ML to classify sentiment.

Scalability: Process large datasets efficiently using PySpark.

Model Evaluation: Metrics include accuracy, F1-score, and ROC AUC.

Visualization: Visualize results with confusion matrix and sentiment distribution.

🔧 Installation

Clone the repository:

git clone https://github.com/omsagvekar/Sentiment-Analysis-on-Movie-Reviews
cd sentiment-analysis-pyspark


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook Main_Copy_of_TWS_Mini_Project.ipynb

📊 Workflow

Data Loading: Load IMDB dataset (50,000+ reviews) and convert it to a PySpark DataFrame for parallel processing.

Preprocessing: Clean and tokenize the reviews while handling any missing values.

Embedding Generation: Use Sentence-BERT to generate semantic embeddings.

Model Training: Build a Logistic Regression model with Spark ML pipeline and split the data into train/test sets.

Evaluation & Visualization: Measure performance with Accuracy, F1-Score, and ROC AUC. Visualize confusion matrix and sentiment distribution.

📈 Model Performance

Accuracy: ~88%

F1-Score: ~0.87

ROC AUC: ~0.90

(Scores may vary depending on dataset sampling and hyperparameters.)

🧩 Key Insights

Sentence-BERT embeddings outperform traditional TF-IDF in capturing contextual meaning.

PySpark enables scalable processing of large-scale text data.

Logistic Regression offers a simple yet effective classification model with interpretable results.

🛠️ Files
Sentiment-Analysis-on-Movie-Reviews/
│
├── Main_Copy_of_TWS_Mini_Project.ipynb     # Main notebook
├── IMDB Dataset.csv                        # IMDB dataset (or provide Kaggle link)
├── README.md                               # Project documentation
└── requirements.txt                        # List of dependencies

📚 Future Enhancements

Integrate Power BI or Streamlit for interactive dashboards.

Add a real-time sentiment API using Flask or FastAPI.

Experiment with DistilBERT and RoBERTa for further improvements in performance.

📄 License

This project is licensed under the MIT License.