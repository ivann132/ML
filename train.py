import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Constants
keypoints_per_frame = 63  # 21 keypoints * 3 dimensions
DATA_PATH = 'MP_Data1'
actions = np.array(['A', 'Absen', 'akhir', 'apung', 'awal', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
no_sequences = 30
sequence_length = 30

# Load Data
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            filepath = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if not os.path.exists(filepath):
                continue  # Skip if file does not exist
            res = np.load(filepath)
            if res.shape[0] != keypoints_per_frame:
                if res.shape[0] > keypoints_per_frame:
                    res = res[:keypoints_per_frame]  # Trim if too long
                else:
                    res = np.pad(res, (0, keypoints_per_frame - res.shape[0]), 'constant')  # Pad if too short
            window.append(res.flatten())
        sequences.append(window)
        labels.append(label_map[action])

# Convert to numpy array and reshape
X = np.array(sequences).reshape(-1, sequence_length, keypoints_per_frame, 1)  # Added extra dimension for channels
y = to_categorical(labels, num_classes=len(actions))

# Train/validation/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Print out the shapes of the datasets
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(sequence_length, keypoints_per_frame, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(actions), activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with validation and early stopping
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluate the model
eval_result = model.evaluate(X_test, y_test)
print(f"Test Loss: {eval_result[0]}")
print(f"Test Accuracy: {eval_result[1]}")

# Save the model
model.save('final_model.h5')

# Save the model architecture as JSON
model_json = model.to_json()
with open('final_model.json', 'w') as json_file:
    json_file.write(model_json)
print("Model architecture saved as JSON.")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

