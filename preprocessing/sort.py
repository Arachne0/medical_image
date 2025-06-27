import os
import pandas as pd
import shutil
import json
from sklearn.preprocessing import MultiLabelBinarizer

data_dir = '/media/hail/HDD/DataSets/NIH_Chest_Xray/' # output
image_root_dir = '/media/hail/HDD/DataSets/NIH_Chest_Xray/original/chest_xray/.' # input
train_val_list = '/media/hail/HDD/DataSets/NIH_Chest_Xray/original/chest_xray/train_val_list.txt'
test_list = '/media/hail/HDD/DataSets/NIH_Chest_Xray/original/chest_xray/test_list.txt'
label_file = '/media/hail/HDD/NIH_Chest_Xray/chest_xray/Data_Entry_2017.csv'
train_label_file = f'{data_dir}/train_labels.json'
test_label_file = f'{data_dir}/test_labels.json'

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

df = pd.read_csv(label_file)
df['Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
unique_labels = set(label for sublist in df['Labels'] for label in sublist if label != 'No Finding')
mlb = MultiLabelBinarizer(classes=list(unique_labels))
multi_hot_labels = mlb.fit_transform(df['Labels'])

label_df = pd.DataFrame(multi_hot_labels, columns=mlb.classes_)
label_df.insert(0, 'Image Index', df['Image Index'])

with open(train_val_list, 'r') as f:
    train_val_images = set(f.read().splitlines())

with open(test_list, 'r') as f:
    test_images = set(f.read().splitlines())

train_labels = {}
test_labels = {}

for _, row in label_df.iterrows():
    image_name = row['Image Index']
    labels = row[1:].to_dict()

    if image_name in train_val_images:
        for root, dirs, files in os.walk(image_root_dir):
            if image_name in files:
                full_path = os.path.join(root, image_name)
                shutil.copy(full_path, os.path.join(train_dir, image_name))
                train_labels[image_name] = labels
                break

    elif image_name in test_images:
        for root, dirs, files in os.walk(image_root_dir):
            if image_name in files:
                full_path = os.path.join(root, image_name)
                shutil.copy(full_path, os.path.join(test_dir, image_name))
                test_labels[image_name] = labels
                break

with open(train_label_file, 'w') as f:
    json.dump(train_labels, f, indent=4)

with open(test_label_file, 'w') as f:
    json.dump(test_labels, f, indent=4)

print(f"Train images and labels saved to '{train_dir}' and '{train_label_file}'")
print(f"Test images and labels saved to '{test_dir}' and '{test_label_file}'")