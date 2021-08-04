import cv2
import numpy as np
from skimage.segmentation import clear_border
from sklearn.cluster import KMeans


# Suxing Liu's method
def segment(image, clusters):
    if clusters < 2: print(f"Must use at least 2 clusters.")

    # flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels)
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # k-means clustering
    kmeans = KMeans(n_clusters=clusters, n_init=40, max_iter=500).fit(reshaped)

    # get labels
    pred_label = kmeans.labels_

    # reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1)
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # sort cluster labels in order of frequency with which they occur
    sortedLabels = sorted([n for n in range(clusters)], key=lambda x: -np.sum(clustering == x))

    # set pixel colors based on clustering
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (clusters - 1)) * i

    ret, thresh = cv2.threshold(kmeansImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh_cleaned = clear_border(thresh)
    return thresh_cleaned
