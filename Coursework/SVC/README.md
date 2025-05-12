# Raisin Classification with SVC
1) Read the data.
2) Convert int64 to float64 (not required but is more organized).
3) Encode label to 1 or 0 (binary classification).
4) Perform train test split so we can see how well the model performed.
5) Perform grid search to find the best parameters for SVC.
6) Display confusion matrix.
## Dataset
- Dataset taken from Kaggle.
- Contains 900 rows of raisin data (50% Kecimen, 50% Besni).
- Images of Kecimen and Besni raisin varieties grown in Turkey.
- Images were preprocessed and features were extracted and converted into quantitive morphological data.
### Columns
```
           Area - The number of pixels within the boundaries of the raisin (these are photos of raisins)
MajorAxisLength - Length of the longest axis in pixels
MinorAxisLength - Length of the shortest axis in pixels
   Eccentricity - Ranges from 0 to 1; 0 is a circle, 1 is highly elongated
     ConvexArea - The area in pixels of the smallest convex shape that encapsulates the raisin
         Extent - Ranges from 0 to 1; Ratio of Area to Bounding Box
      Perimeter - In pixels
          Class - Type of raisin (Kecimen/Besni)
```
## Results
- The models accuracy is about 85% which is pretty decent (correct / total).
  - Since there is even number of each label this is a good measure of evaluation for our model. If it was unbalanced we should consider using precision (punishes false positives), recall (punishes false negatives), or f1 (combination of the two).
![alt text](Figure_1.png)