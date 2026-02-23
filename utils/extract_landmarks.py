import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

#extract 468 landmarks (x, y, z) from image
def extract_landmarks(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Can not read: {img_path}")
        return None
    
    #Convert image from BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Track the face in the image
    res = face_mesh.process(rgb_img)
    #Validate the face detection
    if not res.multi_face_landmarks:
        print(f"Can not detect face in: {res}")
        return None
    
    landmarks = []
    #Extract 468 landmark with standard cordinate (0-1)
    for lm in res.multi_face_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks)

#Extract landmarks and save to scv file
def extract_and_save(base_dir, data_type):
    set_dir = os.path.join(base_dir, data_type)
    #if can not file source path
    if not os.path.exists(set_dir):
        print(f"Can not find directory: {set_dir}")
        return
    
    data = []   #for vector landmarks
    labels = [] #for label

    print(f"\n=============================================")
    print(f"Data Processing: {data_type.upper()}")
    print(f"=============================================")

    #Scan list of faceshape dir


    for label_name in os.listdir(set_dir):
        label_path = os.path.join(set_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        print(f"Folder Processing: {label_name}")

        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)

            #Call extract landmarks func
            feats = extract_landmarks(img_path)

            if feats is not None:
                data.append(feats)
                labels.append(label_name)

    columns = [f"{axis}_{i}" for i in range(468) for axis in ["x", "y", "z"]]
    df = pd.DataFrame(data, columns = columns)
    df["label"] = labels

    OUTPUT_FOLDER = f"processed_data"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    output_filename = os.path.join(OUTPUT_FOLDER, f"raw_landmarks_{data_type}.csv")

    df.to_csv(output_filename, index= False)
    print(f"Saved raw landmarks data into {output_filename}")

if __name__ == "__main__":
    base_dir = "FaceShape Dataset"

    print("Start processing data...")

    extract_and_save(base_dir, "training_set")
    extract_and_save(base_dir, "testing_set")
    print("Done converting img file to csv file!")