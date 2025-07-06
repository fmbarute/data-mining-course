from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load pretrained model
model = ResNet50(weights='imagenet')


def classify_image(img_path):
    """Classify an image using ResNet50 model"""
    try:
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        return decode_predictions(preds, top=5)[0]
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None


def main():
    # Example usage - modify these paths to point to your actual images
    image_dir = os.path.join(os.path.dirname(__file__), 'images')  # Assuming images are in an 'images' subfolder
    image_paths = [
        os.path.join(image_dir, 'animal1.jpg'),
        os.path.join(image_dir, 'animal2.jpg')
        # Add more images as needed
    ]

    # Create directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    for img_path in image_paths:
        print(f"\nPredictions for {img_path}:")
        predictions = classify_image(img_path)

        if predictions:
            for i, (imagenet_id, label, prob) in enumerate(predictions):
                print(f"{i + 1}: {label} ({prob:.2f})")
        else:
            print("Could not process image.")


if __name__ == "__main__":
    main()