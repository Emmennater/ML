# Image Compression with Clustering
1) Read the image.
2) Convert image to n by 3 array (n pixels by 3 color channels).
3) Make KMeans object with number of colors as the number of clusters.
4) Fit KMeans to the pixel data.
5) Predict the cluster for each label using KMeans and assign the new color.
5) Create a new image with the new colors.
## Before
![alt text](splash.jpg)
## After
![alt text](clustered.jpg)