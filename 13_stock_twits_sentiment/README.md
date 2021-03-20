# Stock Twits Sentiment Analysis

## Project
Perform multiclass sentiment analysis to more than a million already labeled stock-twits.

## Models
- Logistic regression for a base model
- Tree based models: Random forest, XGBoost
- Neural Net: LSTM, CONV1D

## Results

| Method | Accuracy | Precision  | Recall |
| ------- | --- | --- | --- |
| LR (BOW)      | 60% | 67% | 63% |
| LR (TFIDF)    | 62% | 70% | 69% |
| XGBoost (BOW) | 62% | 75% | 59% |
| XGBoost (TFIDF)| 63% | 76% | 65% |
| Neural Net (Dense)| 60% | 67% | 66% |
| Neural Net (LSTM)| 60% | 66% | 65% |


## Conclusion

Neural Nets (As fancy as they may sound) don't always work better than simple LR or ensemble methods.

