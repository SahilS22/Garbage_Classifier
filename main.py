import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define CNN model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Load and preprocess dataset
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\sahil\PycharmProjects\pythonProject2\train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\sahil\PycharmProjects\pythonProject2\test_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# Create and train the model
model = create_model((150, 150, 3))
model.fit(
    train_generator,
    steps_per_epoch=2000 // 32,
    epochs=10,
    validation_data=test_generator,
    validation_steps=800 // 32)


# Function to classify garbage or non-garbage from image
def classify_garbage(frame):
    # Preprocess the frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (150, 150))
    frame = frame / 255.0  # Normalize pixel values to be between 0 and 1
    frame = np.reshape(frame, (1, 150, 150, 3))  # Reshape to match model input shape

    # Predict the class (0 for non-garbage, 1 for garbage)
    prediction = model.predict(frame)

    return prediction


# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Couldn't open camera.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame from camera.")
            break

        # Classify garbage in the frame
        prediction = classify_garbage(frame)

        # Display the resulting frame
        if prediction > 0.5:
            cv2.putText(frame, "Garbage Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Garbage Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Garbage Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
