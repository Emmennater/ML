from sklearn.cluster import KMeans
import cv2
import numpy as np

image = cv2.imread('splash.jpg')

# N x 3 matrix
px = image.reshape((-1, 3))
px = np.float32(px)

# Number of colors
clusters = 32
kmeans = KMeans(clusters)
kmeans.fit(px)

# Get clusters / new pixel colors
centers = kmeans.cluster_centers_
labels = kmeans.predict(px)

# Convert back to correct matrix shape
px2 = centers[labels]
px2 = np.array(px2, dtype=np.uint8)
image2 = px2.reshape(image.shape)

# cv2.imwrite('clustered.jpg', image2)
cv2.imshow('orig', image)
cv2.imshow('clust', image2)
cv2.waitKey(0)