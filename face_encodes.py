import os
import cv2
import face_recognition
from mediapipe_face_detect import face_detect
import pickle

#os.chdir("./dataset")
os.chdir("./Sports_data")
save_folder = "D:/Image_clustering/Sports_data_clustering/dataset_sports_faces/"

img_count = 1
encodes = []
for file in os.listdir(os.getcwd()):
    
    img = cv2.imread(file)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_detect(img)

    if boxes is None:
        continue
    
    encodings = face_recognition.face_encodings(img, boxes)
    
    file_name = file.split('.')[0]
    count = 1
    
    for (box, enc) in zip(boxes, encodings):
        top,right,bottom,left = box
        face = img[top:bottom,left:right]
        
        dest_img = save_folder+file_name+"_"+str(count)+'.jpg'
        print(dest_img)
        
        if(face.shape[0]*face.shape[1]*face.shape[2] == 0):
            continue
        cv2.imwrite(dest_img,face)
        d = {"imagePath": dest_img, "loc": box, "encoding": enc}
        encodes.append(d)
        count = count+1
    print(img_count)
    img_count = img_count+1

f = open("D:/Image_clustering/Sports_data_clustering/encodes_sports_data.db", "wb")
f.write(pickle.dumps(encodes))
f.close()
