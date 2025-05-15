# Crop Recommendation Using Decision Trees
- Here is our [Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle.
  - I converted the N, P, K ratios to percents before using them in the model.
- The goal of this project was to use environmental data to identify the crop that would grow the best.
- 40% of the data was used for training the model.
- A decision tree classifier was used to predict the best plant.
- Grid search was in charge of finding the best `depth` and `class_weight` (balanced or none). Balanced class weight means if there are less of that label they will have a higher weight in the predictions.
- The results are very different after each time the program is executed (train test split shuffling).
## Data Fields
- `n` - percent of Nitrogen content in soil
- `p` - percent of Phosphorous content in soil
- `k` - percent of Potassium content in soil
- `temperature` - temperature in degree Celsius
- `humidity` - relative humidity in %
- `ph` - ph value of the soil
- `rainfall` - rainfall in mm
## Results
- Train Score is 1.000
- Test Score is 0.972 (suprisingly good for 15 depth)
- Max depth was 15 (possible overfitting)
## Future Ideas
- A larger dataset might help the model discover broader patterns.
- I tried lowering the max depth manually to 10 and it still did very well.
  - ~0.95 accuracy on train and test.
## Decision Tree Diagram
<img src="DescisionTree.png">
