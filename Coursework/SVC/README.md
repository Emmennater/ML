# Raisin Classification with SVC
1) Read the data.
2) Convert int64 to float64 (not required but is more organized).
3) Encode label to 1 or 0 (binary classification).
4) Perform train test split so we can see how well the model performed.
5) Perform grid search to find the best parameters for SVC.
6) Display confusion matrix.
## Results
- The models accuracy is about 85% which is pretty decent (correct / total).
  - Since there is even number of each label this is a good measure of evaluation for our model. If it was unbalanced we should consider using precision (punishes false positives), recall (punishes false negatives), or f1 (combination of the two).
![alt text](Figure_1.png)