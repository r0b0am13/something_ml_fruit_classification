import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# List of image file paths and corresponding true labels
image_paths = ['t_shirt.png', 'kitten.png', 'dog.png', 'car.png']
true_labels = [1, 2, 3, 4]  # Replace with correct class indices (for evaluation purposes)

# Placeholder for predicted labels
predicted_labels = []

# Process each image
for img_path in image_paths:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict the class
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Top prediction
    
    # Print prediction
    print(f"{img_path} Prediction: {decoded_predictions}")
    
    # Extract and store the predicted class index
    predicted_labels.append(decoded_predictions[0][1])  # Class label

# Example placeholders for true and predicted labels (replace with real labels)
# Let's assume true_labels = [1, 2, 3, 4] corresponding to t-shirt, kitten, dog, car
y_true = [1, 2, 3, 4]  # Replace with correct values
y_pred = predicted_labels  # Convert predicted class labels to their index

# Convert predicted labels to indices if using custom logic
# Example: Label mapping {'tshirt': 1, 'kitten': 2, 'dog': 3, 'car': 4}

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
