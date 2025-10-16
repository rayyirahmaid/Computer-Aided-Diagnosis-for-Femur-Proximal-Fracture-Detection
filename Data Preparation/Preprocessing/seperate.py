import os
import shutil
import random
import yaml

def load_classes_from_yaml(yaml_file):
    """Load class names from the YAML file"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    classes = data['names']
    return classes

def count_classes_in_dataset(labels_dir, classes):
    """Count the number of objects per class in a given labels directory"""
    class_count = {cls: 0 for cls in classes}  # Initialize class counts to 0
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            # Read all lines in the label file, each line corresponds to an object in the image
            for line in f:
                # The first value is the class ID
                class_id = int(line.split()[0])  # Get class ID from the first value in the line
                if class_id < len(classes):  # Check if class_id is valid
                    class_name = classes[class_id]
                    class_count[class_name] += 1  # Increment the count for this class

    return class_count

def count_classes_in_train_test(train_labels_dir, test_labels_dir, yaml_file):
    """Count the number of objects per class in both the train and test datasets"""
    classes = load_classes_from_yaml(yaml_file)
    
    # Count classes in train dataset
    print("Counting objects in train dataset...")
    train_class_count = count_classes_in_dataset(train_labels_dir, classes)
    print("Train class counts:", train_class_count)
    
    # Count classes in test dataset
    print("Counting objects in test dataset...")
    test_class_count = count_classes_in_dataset(test_labels_dir, classes)
    print("Test class counts:", test_class_count)

def split_dataset(dataset_dir, train_dir, test_dir, test_size=0.2):
    # Create train and test directories if they don't exist
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

    # Separate images and labels
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    # Get image files (supporting .jpg, .png, .jpeg)
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Determine the number of test images
    num_test = int(len(image_files) * test_size)
    test_files = image_files[:num_test]
    train_files = image_files[num_test:]

    # Copy training files to the train directories
    for file in train_files:
        img_path = os.path.join(images_dir, file)
        label_path = os.path.join(labels_dir, file.replace(file.split('.')[-1], 'txt'))  # Replace image extension with .txt

        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(train_dir, 'images', file))  # Copy image

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(train_dir, 'labels', file.replace(file.split('.')[-1], 'txt')))  # Copy label
        else:
            print(f"Warning: Label file not found for {file}.")

    # Copy testing files to the test directories
    for file in test_files:
        img_path = os.path.join(images_dir, file)
        label_path = os.path.join(labels_dir, file.replace(file.split('.')[-1], 'txt'))  # Replace image extension with .txt

        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(test_dir, 'images', file))  # Copy image

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(test_dir, 'labels', file.replace(file.split('.')[-1], 'txt')))  # Copy label
        else:
            print(f"Warning: Label file not found for {file}.")

    print(f"Dataset split: {len(train_files)} training images, {len(test_files)} testing images")

# Example usage
if __name__ == '__main__':
    processed_dir = 'C:/RayFile/KuliahBro/Semester8/TugasAkhir/Wrist/clahe_dataset'
    train_dir = 'C:/RayFile/KuliahBro/Semester8/TugasAkhir/Wrist/dataset/train'
    test_dir = 'C:/RayFile/KuliahBro/Semester8/TugasAkhir/Wrist/dataset/test'

    # Then split the processed dataset
    split_dataset(processed_dir, train_dir, test_dir)
    
    yaml_file = 'C:/RayFile/KuliahBro/Semester8/TugasAkhir/Wrist/dataset/meta.yaml'  # Path to your YAML file
    train_labels_dir = 'C:/RayFile/KuliahBro/Semester8/TugasAkhir/Wrist/dataset/train/labels'
    test_labels_dir = 'C:/RayFile/KuliahBro/Semester8/TugasAkhir/Wrist/dataset/test/labels'
    
    count_classes_in_train_test(train_labels_dir, test_labels_dir, yaml_file)