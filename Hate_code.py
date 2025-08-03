import pandas as pd
import seaborn as sns
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
import pickle
import keras

# Load datasets
imbalance_data = pd.read_csv(r"data\imbalanced_data.csv")
raw_data = pd.read_csv(r"data\raw_data.csv")

# Clean and prepare raw_data
del_cols = ['Unnamed: 0','count','hate_speech','offensive_language','neither']
raw_data.drop(del_cols, axis=1, inplace=True)
raw_data[raw_data['class']==0]['class'] = 1
raw_data["class"].replace({0: 1, 2: 0}, inplace=True)
raw_data.rename(columns={'class': 'label'}, inplace=True)

# Merge both datasets
imbalance_data.drop("id", axis=1, inplace=True)
df = pd.concat([imbalance_data, raw_data])

# Download stopwords
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Data cleaning function
def data_cleaning(words):
    words = str(words).lower()
    words = re.sub(r'\[.*?\]', '', words)
    words = re.sub(r'https?://\S+|www\.\S+', '', words)
    words = re.sub(r'<.*?>+', '', words)
    words = re.sub(r'[%s]' % re.escape(string.punctuation), '', words)
    words = re.sub(r'\n', '', words)
    words = re.sub(r'\w*\d\w*', '', words)
    words = [word for word in words.split() if word not in stopword]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# Apply cleaning
df['tweet'] = df['tweet'].apply(data_cleaning)

# Split data
x = df['tweet']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Tokenization and padding
max_words = 50000
max_len = 300
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
sequences_matrix = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len)

# Build model
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
model.fit(sequences_matrix, y_train, batch_size=128, epochs=1, validation_split=0.2)

# Evaluate
test_sequences_matrix = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=max_len)
accr = model.evaluate(test_sequences_matrix, y_test)
lstm_prediction = model.predict(test_sequences_matrix)
res = [1 if pred[0] >= 0.5 else 0 for pred in lstm_prediction]
print(confusion_matrix(y_test, res))

# Save model and tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model.save("model.h5")

# Load and test on sample
load_model = keras.models.load_model("model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

test_input = 'cause Im tired of you big bitches coming for us skinny girls'
test_input = [clean_text(test_input)]
seq = load_tokenizer.texts_to_sequences(test_input)
padded = pad_sequences(seq, maxlen=300)
pred = load_model.predict(padded)
print("Prediction:", "hate and abusive" if pred >= 0.5 else "no hate")
