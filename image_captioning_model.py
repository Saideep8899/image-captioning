import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Step 1: Load VGG16 for feature extraction
def extract_features(image_path):
    model = VGG16(include_top=False, weights='imagenet')
    model = Sequential(model.layers[:-1])  # Remove top layer
    img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for VGG16

    features = model.predict(img_array)
    features = features.flatten()  # Flatten the feature vector
    return features

# Step 2: Tokenize captions (Text Preprocessing)
def preprocess_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max([len(caption.split()) for caption in captions])
    return tokenizer, vocab_size, max_length

# Step 3: Define the LSTM-based Captioning Model
def define_captioning_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Step 4: Generate Caption from Image Features
def generate_caption(model, image_features, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Example Usage
image_path = 'example_image.jpg'  # Path to your image
captions = ['a dog is playing with a ball', 'a dog is running in the park']  # Example captions
tokenizer, vocab_size, max_length = preprocess_captions(captions)

# Extract features from the image
image_features = extract_features(image_path)

# Define the model and train (omitting training code for simplicity)
model = define_captioning_model(vocab_size, max_length)

# Generate caption
caption = generate_caption(model, image_features, tokenizer, max_length)
print("Generated Caption:", caption)
