import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─── SETTINGS ───────────────────────────────────────────
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 43
DATASET_PATH = "dataset"

# ─── STEP 1: DATA LOAD KARO ─────────────────────────────
print("Data load ho raha hai...")

train_df = pd.read_csv(os.path.join(DATASET_PATH, "Train.csv"))
test_df  = pd.read_csv(os.path.join(DATASET_PATH, "Test.csv"))

def load_images(df, base_path, img_size):
    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row["Path"])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(row["ClassId"])
    return np.array(images), np.array(labels)

X_train, y_train = load_images(train_df, DATASET_PATH, IMG_SIZE)
X_test,  y_test  = load_images(test_df,  DATASET_PATH, IMG_SIZE)

print(f"Training images: {X_train.shape}")
print(f"Testing images:  {X_test.shape}")

# ─── STEP 2: PREPROCESSING ──────────────────────────────
print("Preprocessing ho raha hai...")

# ResNet50 ka apna preprocessing
X_train = preprocess_input(X_train.astype(np.float32))
X_test  = preprocess_input(X_test.astype(np.float32))

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

# Train/Val split manually
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_cat, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

# ─── STEP 3: RESNET50 MODEL BANAO ───────────────────────
print("ResNet50 Model ban raha hai...")

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── STEP 4: PHASE 1 TRAIN ──────────────────────────────
print("\nPhase 1: Top layers train ho rahi hain...")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

history1 = model.fit(
    X_tr, y_tr,
    batch_size=BATCH_SIZE,
    epochs=8,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ─── STEP 5: PHASE 2 FINE TUNING ────────────────────────
print("\nPhase 2: Fine tuning shuru ho rahi hai...")

for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    X_tr, y_tr,
    batch_size=BATCH_SIZE,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ─── STEP 6: EVALUATE ───────────────────────────────────
print("\nEvaluation ho rahi hai...")

y_pred_prob = model.predict(X_test)
y_pred      = np.argmax(y_pred_prob, axis=1)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n===== ResNet50 Results =====")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")

# ─── STEP 7: GRAPHS SAVE ────────────────────────────────
os.makedirs("results", exist_ok=True)

combined_acc      = history1.history['accuracy']     + history2.history['accuracy']
combined_val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
combined_loss     = history1.history['loss']         + history2.history['loss']
combined_val_loss = history1.history['val_loss']     + history2.history['val_loss']

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(combined_acc,     label='Train Accuracy')
plt.plot(combined_val_acc, label='Val Accuracy')
plt.title('ResNet50 - Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(combined_loss,     label='Train Loss')
plt.plot(combined_val_loss, label='Val Loss')
plt.title('ResNet50 - Loss')
plt.legend()

plt.savefig("results/resnet_accuracy_loss.png")
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15,15))
sns.heatmap(cm, annot=False, fmt='d', cmap='Greens')
plt.title('ResNet50 - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("results/resnet_confusion_matrix.png")
plt.show()

model.save("models/resnet_model.h5")
print("Model save ho gaya: models/resnet_model.h5")

resnet_metrics = {
    "accuracy":  acc,
    "precision": prec,
    "recall":    rec,
    "f1_score":  f1
}
with open("models/resnet_metrics.pkl", "wb") as f:
    pickle.dump(resnet_metrics, f)

print("\nResNet50 ka kaam mukammal! 🎉")