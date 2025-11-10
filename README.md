# Sentiment Analysis on Movie Reviews

🎬 **Sentiment Analysis on Movie Reviews** is an end-to-end sentiment classification project using **PySpark** and **Sentence-BERT**.  
It combines the scalability of **Big Data (PySpark)** with the contextual accuracy of **Transformer-based embeddings (Sentence-BERT)** to predict whether a movie review expresses a **positive** or **negative** sentiment.

## 🚀 Features

- **Preprocessing**: Clean, tokenize, and handle missing values in movie reviews.  
- **Sentence-BERT**: Generate semantic embeddings for better context understanding.  
- **Logistic Regression**: Classify sentiments using **Spark ML**.  
- **Scalability**: Efficiently handle large datasets using **PySpark**.  
- **Model Evaluation**: Accuracy, F1-score, and ROC AUC.  
- **Visualization**: Visualize confusion matrix and sentiment distribution.

## 🔧 Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/omsagvekar/Sentiment-Analysis-on-Movie-Reviews
    cd sentiment-analysis-pyspark
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the notebook**:

    ```bash
    jupyter notebook Main_Copy_of_TWS_Mini_Project.ipynb
    ```

## 🎬 How to Use

1. **Data Loading**: Load the IMDB dataset (50,000+ reviews) and convert it to a **PySpark DataFrame**.  
2. **Preprocessing**: Clean and tokenize the text, handle missing values.  
3. **Embedding Generation**: Generate embeddings using **Sentence-BERT**.  
4. **Model Training**: Train a **Logistic Regression** model using the Spark ML pipeline.  
5. **Evaluation & Visualization**: Evaluate accuracy, F1-score, and ROC AUC, and visualize the results.

## 📈 Model Performance

- **Accuracy**: ~88%  
- **F1-Score**: ~0.87  
- **ROC AUC**: ~0.90  

*(Scores may vary depending on dataset sampling and hyperparameters.)*

## 🛠️ Files

Sentiment-Analysis-on-Movie-Reviews/  
│  
├── Main_Copy_of_TWS_Mini_Project.ipynb   # Main notebook  
├── IMDB Dataset.csv                      # IMDB dataset (or Kaggle link)  
├── README.md                             # Project documentation  
└── requirements.txt                      # Dependencies  

## 🧩 Key Insights

- **Sentence-BERT** embeddings outperform traditional TF-IDF by capturing contextual meaning.  
- **PySpark** enables scalable processing of large-scale text data.  
- **Logistic Regression** provides interpretability with consistent performance.  

## 📚 Future Enhancements

- Integrate **Power BI** or **Streamlit** for interactive dashboards.  
- Add a **real-time sentiment API** using **Flask** or **FastAPI**.  
- Experiment with **DistilBERT** and **RoBERTa** for improved contextual understanding.  

## 👨‍💻 Author

**Om Sunil Sagvekar**  
📧 omsagvekar04@gmail.com  
🔗 LinkedIn: [https://www.linkedin.com/in/om-sagvekar](https://www.linkedin.com/in/om-sagvekar)  
🔗 GitHub: [https://github.com/omsagvekar](https://github.com/omsagvekar)

## 📄 License

This project is licensed under the **MIT License**.
