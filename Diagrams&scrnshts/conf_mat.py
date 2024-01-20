import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

img_size = 48
batch_size = 64
base_path = "C:\\Users\\arvin\\OneDrive\\Desktop\\miniproj\\"

# Load the saved model architecture from JSON file
with open('model.json', 'r') as json_file:
    json_savedModel = json_file.read()

# Load the model architecture from the JSON file
model = model_from_json(json_savedModel)
model.load_weights('model_weights.h5')

# Data generator for testing
datagen_test = ImageDataGenerator()
test_generator = datagen_test.flow_from_directory(base_path + "validation",
                                                  target_size=(img_size, img_size),
                                                  color_mode='grayscale',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Use the model to predict the class probabilities for the test data
y_pred = model.predict(test_generator)

# Convert the predicted class probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true class labels for the test data
y_true = test_generator.classes

# Get the class labels for the dataset
class_labels = list(test_generator.class_indices.keys())

# Print the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
plt.figure(figsize=(10, 10))
sns.heatmap(df_cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
