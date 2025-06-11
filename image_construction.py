import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split


# function to create a folder
def create_folder(folder_name):
    try:
        os.makedirs(folder_name, exist_ok=True)
        print(f"Folder '{folder_name}' created successfully.")
    except OSError as e:
        print(f"Error creating folder '{folder_name}': {e}")


#Read image data as pixels and label, create a folder for each label 
def read_and_create_folders(filepath):
    df = pd.read_csv(filepath)
    print("------Creating folders------")
    if 'label' in df.columns:
        create_folder("Data/train/")
        create_folder("Data/validation/")
        labels = set(np.array(df['label']))
        for label in labels:
            create_folder(f'Data/train/{label}')
            create_folder(f'Data/validation/{label}')
    else:
        create_folder("Data/test/")
    return df


#construct the 28*28 image one by one
def construct_image(df):
    images_list = df
    if "label" in images_list.columns:
        images_list = df.drop(["label"],axis=1)
    cols = images_list.columns
    images = list()
    for i in range(len(images_list)):
        a = list()
        row = list()
        for j in range(len(cols)):
            if j % 28 == 0 and j != 0:
                a.append(row)
                row = list()
            row.append(images_list.loc[i, cols[j]])
        image = np.array(a, dtype=np.uint8)
        images.append(image)
    return images

#save image in its right label folder
def save_image(images , labels = None , validation = False):
    print("------Saving constructed images------")
    for i in range(len(images)):
        new_image = Image.fromarray(images[i])
        if labels:
            if validation:
                new_image.save(f'Data/validation/{labels[i]}/{i}.png')
            else:
                new_image.save(f'Data/train/{labels[i]}/{i}.png')
        else:
            new_image.save(f'Data/test/{i}.png')


if __name__ == "__main__":
    image_csv = input("Type the path of image csv file")
    print("------Reading csv file------")
    if os.path.exists(image_csv):
        df = read_and_create_folders(image_csv)
    else:
        print("Invalid path. Please check the file.")
    print("------Constructing Images------")
    
    if 'label' in df.columns:

        df_train , df_validation = train_test_split(df,test_size=0.2)

        # Reset their indices
        df_train = df_train.reset_index(drop=True)
        df_validation = df_validation.reset_index(drop=True)

        train_images = construct_image(df_train)
        train_labels = list(df_train["label"])

        validation_images = construct_image(df_validation)
        validation_labels = list(df_validation["label"])

        save_image(train_images,train_labels)
        save_image(validation_images,validation_labels ,validation= True)

    else:
        images = construct_image(df)
        labels = None
        save_image(images,labels)

