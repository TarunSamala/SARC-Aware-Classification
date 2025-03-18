import pandas as pd
import numpy as np
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Concatenate  # Changed LSTM to GRU
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('train-balanced-sarcasm.csv', usecols=['label', 'parent_comment', 'score'])

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text

df['cleaned_text'] = df['parent_comment'].apply(clean_text)

# Prepare text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Prepare numerical data
scaler = StandardScaler()
scaled_scores = scaler.fit_transform(df[['score']])

# Split data
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    padded_sequences, scaled_scores, df['label'], test_size=0.2, random_state=42)

# Build Bi-GRU model with dual inputs
text_input = Input(shape=(100,), name='text_input')
num_input = Input(shape=(1,), name='score_input')

# Text processing branch (GRU instead of LSTM)
x = Embedding(input_dim=10000, output_dim=128)(text_input)
x = Bidirectional(GRU(64, return_sequences=True))(x)  # First GRU layer
x = Bidirectional(GRU(32))(x)                        # Second GRU layer

# Numerical score branch (unchanged)
y = Dense(16, activation='relu')(num_input)

# Concatenate branches
combined = Concatenate()([x, y])

# Classification head
z = Dense(32, activation='relu')(combined)
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[text_input, num_input], outputs=output)

# Compile model (unchanged)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train model (unchanged)
history = model.fit(
    [X_text_train, X_num_train],
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    class_weight={0: 1, 1: 2}
)

# Evaluate (unchanged)
results = model.evaluate([X_text_test, X_num_test], y_test)
print(f"Test Accuracy: {results[1]:.2f}")
print(f"Test Precision: {results[2]:.2f}")
print(f"Test Recall: {results[3]:.2f}")

# Save model
model.save('sarcasm_detection_biGRU.h5')  # Changed filename