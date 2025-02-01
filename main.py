import pandas as pd
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from matplotlib import pyplot as plt


def read_images_and_annotations(annotations, image_dir):
    image_data = []
    for index, row in annotations.iterrows():
        image_name = row["name"]
        image_path = os.path.join(image_dir, "0" + image_name[:3], image_name)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            annotations_dict = {
                "image": image,
                "annotations": {
                    "human": row["human"],
                    "vehicle": row["vehicle"],
                    "property": row["property"],
                    "opening": row["opening"],
                    "plant": row["plant"],
                    "signal": row["signal"],
                    "entry": row["entry"],
                    "furniture": row["furniture"],
                    "seat": row["seat"],
                    "desk": row["desk"],
                    "buttons": row["buttons"],
                    "fore": row["fore"]
                }
            }
            image_data.append(annotations_dict)
        else:
            print(f"Image not found: {image_path}")
    return image_data


def create_dataset(image_data):
    images = []
    annotations = []

    for data in image_data:
        images.append(data["image"])
        annotations.append(data["annotations"])

    # Convert annotations to target tensors
    num_classes = len(annotations[0])  # Number of classes

    target_tensors = []
    for annotation in annotations:
        target_tensor = np.zeros(num_classes)
        for i, (key, value) in enumerate(annotation.items()):
            target_tensor[i] = value
        target_tensors.append(target_tensor)

    # Convert images to arrays
    image_arrays = [img_to_array(img) for img in images]
    image_arrays = np.array(image_arrays)

    # Create TensorFlow Dataset
    return tf.data.Dataset.from_tensor_slices((image_arrays, target_tensors))


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.GlobalMaxPool2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.summary()
    return model


def main():
    print("Reading annotations...")
    annotations_file = pd.read_csv("dataset/annotations.csv")

    print("Reading images...")
    image_data = read_images_and_annotations(annotations_file, "dataset")[:3000]

    print("Creating dataset...")
    dataset = create_dataset(image_data)

    dataset_size = len(image_data)
    train_size = int(0.8 * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    batch_size = 32
    train_dataset = train_dataset.shuffle(buffer_size=train_size).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    print("Creating model...")
    model = create_model((256, 256, 3), 12)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("Training model...")
    h = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

    print("Saving model...")
    model.save("model.keras")

    # Print the validation loss
    print("Validation loss:", np.min(h.history["val_loss"]))
    print("Validation accuracy:", np.max(h.history["val_accuracy"]))

    # Plot the training and validation loss
    plt.plot(h.history["loss"])
    plt.plot(h.history["val_loss"])
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(h.history["accuracy"])
    plt.plot(h.history["val_accuracy"])
    plt.show()


if __name__ == "__main__":
    main()
