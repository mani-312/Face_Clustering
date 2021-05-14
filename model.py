# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("[INFO] Loaded modules")
encode_file = "./encodes_sports_data.db"
data = pickle.loads(open(encode_file, "rb").read())
encodings = [d["encoding"] for d in data]

print("[INFO] loaded data")

# Looking for optimal number clusters
WCSS = []
for i in range(1,15):
    model = KMeans(n_clusters=i)
    model.fit(encodings)
    WCSS.append(model.inertia_)

# Visualizing K vs WCSS
plt.plot(range(1,15),WCSS)
plt.xlabel("K- Number of clusters")
plt.ylabel("WCSS")
plt.show()

print("[INFO] training model")
n_clusters = 5
model = KMeans(n_clusters)
model.fit(encodings)

print("[INFO] Visulaizing output")

for labelID in range(n_clusters):
    idxs = np.where(model.labels_ == labelID)[0]
    faces = []
	# loop over the sampled indexes
    count=0
    for i in idxs:
        # load the input image and extract the face ROI
        img_path = data[i]["imagePath"]
        
        face = cv2.imread(img_path)
       
        count = count+1
        print(count)
        
        # force resize the face ROI to 96x96 and then add it to the
		# faces montage list
        face = cv2.resize(face, (96, 96))
        faces.append(face)
	# create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_mo ntages(faces, (96, 96), (9,9))[0]
	
	# show the output montage
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    cv2.imshow(title, montage)
    cv2.waitKey(0)

class_path = "D:/Image_clustering/Sports_data_clustering/Classes/"
dataset_path = "D:/Image_clustering/Sports_data_clustering/Sports_data/"
print("[INFO] Creating label directories")
for i in range(n_clusters):
    os.mkdir(class_path+"Label_"+str(i))

print("[INFO] Saving the images")
for (img,label) in zip(data,model.labels_):
    name = img["imagePath"].split('/')[-1]
    parent = name.split('.')[0][:-2]
    parent_name = parent+".jpg"
    path = dataset_path+parent_name
    print(label)
    image = cv2.imread(path)
    if image is None:
        continue
    cv2.imwrite(class_path+"Label_"+str(label)+"/"+parent_name,image)
