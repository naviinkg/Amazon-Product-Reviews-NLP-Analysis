# Amazon-Product-Reviews-NLP-Analysis

## Project Overview
This project focuses on analyzing Amazon product reviews using Natural Language Processing (NLP) techniques. The main objectives include topic modeling, sentiment analysis, and classification of customer reviews to extract meaningful insights from textual data.

## Key Features
* Data preprocessing and cleaning of Amazon product reviews
* Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner)
* Topic modeling using Latent Dirichlet Allocation (LDA) or Non-Negative Matrix Factorization (NMF)
* Exploratory Data Analysis (EDA) with visualizations
* Machine learning models for review classification
  
## Technologies Used
* Python
* Pandas for data manipulation
* NLTK for natural language processing tasks
* VaderSentiment for sentiment analysis
* Matplotlib and Seaborn for data visualization
* Scikit-learn for machine learning models
  
## Project Structure
* Data Loading and Preprocessing
* Text Cleaning and Tokenization
* Sentiment Analysis
* Exploratory Data Analysis
* Topic Modeling
* Machine Learning Model Implementation
  
## Installation and Usage
* Clone the repository
* Run the Jupyter notebook: jupyter notebook Amazon_Product_Reviews_NLP_Analysis.ipynb

## Results
* Sentiment distribution analysis of customer reviews
* Word cloud visualizations of review content
* Topic modeling results and interpretations
* Performance metrics of various machine learning models

## Future Work
* Implement advanced NLP techniques for deeper text analysis
* Explore other topic modeling algorithms for comparison
* Develop a web application for real-time review analysis









## Elaborate description of every step of the project:
### a) Data Import and Initial Exploration:

The project begins by importing necessary libraries and loading the dataset from a CSV file.

Initial exploration includes viewing the first and last few rows of the data, checking data info, and examining data types.

### b) Data Cleaning:
Missing values are identified and handled.

Columns with any NaN values are removed to ensure data quality.

### c) Text Preprocessing:
A series of text cleaning functions are applied:

Converting text to lowercase

Removing URLs, markdown-style links, and @username mentions

Removing punctuation and special characters

Tokenization is performed to split text into individual words

Stopwords are removed to focus on meaningful content

Stemming is applied to reduce words to their root form

#### d) Sentiment Analysis:
VADER (Valence Aware Dictionary and sEntiment Reasoner) is used for sentiment analysis

Sentiment scores are calculated for each review

Reviews are classified as Positive, Negative, or Neutral based on compound scores

### e) Exploratory Data Analysis (EDA):
Visualization of sentiment distribution using a bar plot

Creation of word clouds to visualize frequently occurring words in reviews

Separate word clouds are generated for positive, neutral, and negative reviews

## Elaborate analysis and descriptions of the results:

### a) Sentiment Distribution:

The analysis reveals a strong positive bias in the reviews:
* Positive reviews: 1,421
* Neutral reviews: 54
* Negative reviews: 122
* 
This distribution suggests high customer satisfaction but may also indicate potential bias in the data collection process or the need for more balanced sampling.
### b) Word Cloud Analysis:

The general word cloud highlights frequently used terms across all reviews, giving an overview of common themes or topics.

Sentiment-specific word clouds provide insights into the language used in positive, neutral, and negative reviews:

Positive reviews may emphasize product features, ease of use, or customer satisfaction

Negative reviews might highlight issues, complaints, or areas for improvement

Neutral reviews could contain mixed sentiments or factual statements

### c) Implications:
The overwhelmingly positive sentiment suggests that the product (likely Kindle devices, based on the data) is well-received by customers.

The low number of neutral reviews indicates that most customers have strong opinions, either positive or negative.

The small proportion of negative reviews provides valuable feedback for potential product improvements or addressing customer concerns.

## Description of the data:

The dataset appears to be a collection of Amazon product reviews, likely focusing on Kindle devices. Key features of the data include:
* Product information: ID, ASIN (Amazon Standard Identification Number), brand, categories, colors, dimensions, etc.
* Review metadata: Date added, date updated, review date, rating, helpfulness votes, etc.
* Review content: Review text, title, username of reviewer
* Derived features: Cleaned review text, tokenized reviews, stemmed reviews, sentiment scores and classifications
* The data structure allows for comprehensive analysis of customer feedback, including both quantitative (ratings, helpfulness votes) and qualitative (review text) aspects. The preprocessing steps applied to the review text enable advanced natural language processing techniques, such as sentiment analysis and potentially topic modeling (though not explicitly shown in the provided excerpt).
* This rich dataset provides a solid foundation for understanding customer perceptions, identifying common themes in feedback, and potentially informing product development or marketing strategies for Amazon's Kindle line or similar e-reader devices.
