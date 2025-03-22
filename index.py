import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

# 🔹 Step 1: Load the IMDB dataset (Preprocessed as integers)
vocab_size = 20000  # Only keep top 20k words
max_length = 200  # Pad sequences to this length

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 🔹 Step 2: Pad sequences to make all inputs the same length
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# 🔹 Step 3: Build CNN + LSTM model
model = Sequential([
    # 1️⃣ Embedding Layer
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),

    # 2️⃣ Convolution Layer for Feature Extraction
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # 3️⃣ LSTM for Sequential Understanding
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(50),

    # 4️⃣ Fully Connected Layers
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),

    # 5️⃣ Output Layer (Binary Classification)
    Dense(1, activation='sigmoid')  # Use softmax for multi-class classification
])

# 🔹 Step 4: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔹 Step 5: Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# 🔹 Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
