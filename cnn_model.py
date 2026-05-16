import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ─── SETTINGS ───────────────────────────────────────────
IMG_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 43
DATASET_PATH = "dataset"

# ─── STEP 1: DATA LOAD KARO ─────────────────────────────
print("Data load ho raha hai...")

train_df = pd.read_csv(os.path.join(DATASET_PATH, "Train.csv"))
test_df  = pd.read_csv(os.path.join(DATASET_PATH, "Test.csv"))

def load_images(df, base_path):
    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row["Path"])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(row["ClassId"])
    return np.array(images), np.array(labels)

X_train, y_train = load_images(train_df, DATASET_PATH)
X_test,  y_test  = load_images(test_df,  DATASET_PATH)

print(f"Training images: {X_train.shape}")
print(f"Testing images:  {X_test.shape}")

# ─── STEP 2: PREPROCESSING ──────────────────────────────
print("Preprocessing ho raha hai...")

X_train = X_train / 255.0
X_test  = X_test  / 255.0

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

# ─── STEP 3: CNN MODEL BANAO ────────────────────────────
print("CNN Model ban raha hai...")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── STEP 4: TRAIN KARO ─────────────────────────────────
print("Training shuru ho rahi hai...")

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ─── STEP 5: EVALUATE KARO ──────────────────────────────
print("Evaluation ho rahi hai...")

y_pred_prob = model.predict(X_test)
y_pred      = np.argmax(y_pred_prob, axis=1)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')

print(f"\n===== CNN Results =====")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")

# ─── STEP 6: RESULTS SAVE KARO ──────────────────────────
os.makedirs("results", exist_ok=True)

# Accuracy & Loss Curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('CNN - Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CNN - Loss')
plt.legend()

plt.savefig("results/cnn_accuracy_loss.png")
plt.show()
print("Graph save ho gaya: results/cnn_accuracy_loss.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15,15))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('CNN - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("results/cnn_confusion_matrix.png")
plt.show()
print("Confusion Matrix save ho gayi: results/cnn_confusion_matrix.png")

# Model Save Karo
model.save("models/cnn_model.h5")
print("Model save ho gaya: models/cnn_model.h5")

# Metrics Save Karo
cnn_metrics = {
    "accuracy":  acc,
    "precision": prec,
    "recall":    rec,
    "f1_score":  f1
}
with open("models/cnn_metrics.pkl", "wb") as f:
    pickle.dump(cnn_metrics, f)

print("\nCNN ka kaam mukammal! 🎉")