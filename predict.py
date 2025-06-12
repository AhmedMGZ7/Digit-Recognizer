import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import csv
from tqdm import tqdm


# Load the saved model
model = load_model("./models/cnn_digit_recognizer.h5")

# Define class labels
class_labels = ['0','1','2','3','4','5','6','7','8','9']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256), color_mode="rgb")  # RGB mode and correct size
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, 256, 256, 3)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_labels[np.argmax(predictions)]

    return predicted_class

def get_number(filename):
    return int(os.path.splitext(filename)[0])

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing the images: ")

    if os.path.isdir(folder_path):
        results = []

        # Sort image files to maintain consistent order
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = sorted(image_files, key=get_number)
        
        print("---------Reading and Predicting----------")
        for idx, img_name in enumerate(tqdm(image_files, desc="Predicting images", unit="img"), start=1):
            img_path = os.path.join(folder_path, img_name)
            label = predict_image(img_path)
            results.append([idx, label])

        # Save to CSV
        csv_path = os.path.join(folder_path, "predictions.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ImageId", "Label"])
            writer.writerows(results)

        print(f"Predictions saved to {csv_path}")
    else:
        print("Invalid folder path. Please check and try again.")
