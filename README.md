# IMDb Movie Review Sentiment Analysis ( MIC CLUB )

## Objective
Classify IMDb movie reviews as ‘positive’ or ‘negative’ using Logistic Regression (beginner batch).

## Dataset
- Contains 50,000 reviews labeled as positive or negative.

## Approach & Steps

1. **Loading Data:** Read the CSV file containing reviews and sentiment labels.
2. **Text Preprocessing:** Use CountVectorizer to convert review text into numeric features.
3. **Splitting:** Split data into train and test sets (80/20).
4. **Model:** Train Logistic Regression on the vectorized data.
5. **Evaluation:** Check accuracy and show a classification report.

## Result
Test accuracy achieved: 87%
## Dependencies
- Python 3.x
- pandas
- scikit-learn

## How to Run
1. Download the dataset CSV and place it in the same folder.
2. Run:
