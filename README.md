# AI/ML Engineering Internship - DevelopersHub Corporation

This project is a part of my **AI/ML Engineering Internship** at **DevelopersHub Corporation**, Islamabad.

## Internship Details

- **Company:** DevelopersHub Corporation, Islamabad 🇵🇰
- **Internship Period:** July - September 2025

# News Topic Classifier with BERT

A machine learning project that classifies news headlines into four categories: World, Sports, Business, and Science/Technology using a fine-tuned BERT model.

- Video Demo of app - https://youtu.be/zsoRqG3eiSw
> **Note:** In the demo, I used the dataset from [DataCamp](https://www.datacamp.com/datalab/datasets/dataset-r-news-articles). Before using it in the app, I replaced the `title` column with `text`.

## Objective

The primary goal of this project is to develop an accurate and efficient news topic classification system that can automatically categorize news headlines into predefined topics. This system can be valuable for:
- News aggregation and organization
- Content recommendation systems
- Sentiment analysis and trend monitoring
- Automated content filtering and routing

## Methodology / Approach

### Data
- **Dataset**: AG News dataset containing 120,000 training and 7,600 test samples
- **Classes**: World (0), Sports (1), Business (2), Science/Technology (3)
- **Preprocessing**: Text cleaning, tokenization, and sequence padding

### Model Architecture
- **Base Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Fine-tuning**: Custom classification head for 4 news categories
- **Training**: Transfer learning with Adam optimizer and learning rate scheduling

### Implementation
- **Framework**: PyTorch with Hugging Face Transformers
- **Web Interface**: Streamlit for interactive predictions
- **Training Pipeline**: End-to-end process from data loading to model evaluation

## Key Results

- Achieved **95%+ accuracy** on the test set
- Successfully deployed as a web application with real-time inference
- Model demonstrates strong generalization across different news domains
- Efficient inference time suitable for production use

## Features

- **Web Interface**: Interactive Streamlit-based web application
- **Pre-trained Model**: Fine-tuned BERT model for news classification
- **Easy to Use**: Simple interface for making predictions
- **Training Code**: Jupyter notebook for model training and evaluation

## Installation

1. Clone the repository:
   ```bash
   git clone 'https://github.com/maryamzaman30/Data-Driven-Personalized-Educational-Content-Recommendation-System.git'
   cd NewsTopicBERT
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv news-env
   source news-env/bin/activate  # On Windows: news-env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Setup

This project uses a pre-trained BERT model for news classification. Follow these steps to set up the model:

1. **Download the model files**:
   - The model files are not included in the repository due to their size
   - you can train the model yourself using the provided Jupyter notebook: `news_classifier_training.ipynb`

2. **Directory Structure**:
   After downloading, place the model files in the following structure:
   ```
   NewsTopicBERT/
   └── fine_tuned_bert_agnews/
       ├── config.json
       ├── metadata.joblib
       ├── special_tokens_map.json
       ├── tokenizer_config.json
       ├── tokenizer.json
       └── vocab.txt
   ```

3. **Verify Installation**:
   The application will automatically load the model from the `fine_tuned_bert_agnews/` directory when you start the app.

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter a news headline or select a sample to see the classification results

4. Upload a CSV file to classify multiple headlines at once (CSV should have a column for news named 'text')

## Project Structure

```
NewsTopicBERT/
├── app.py                # Streamlit web application
├── model_utils.py        # Model loading and prediction utilities
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── news_classifier_training.ipynb  # Model training notebook
```

## Acknowledgments

- BERT by Google Research
- Hugging Face Transformers
- AG News Dataset
