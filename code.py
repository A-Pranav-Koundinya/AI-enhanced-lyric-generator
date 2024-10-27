import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
# Initialize the Tokenizer
tokenizer = Tokenizer()
# Path to your file
file_path = "aip lyrics.txt"
# Read data from the file
with open(file_path, 'r') as file:
data = file.read()
# Convert 'data' to lowercase and split into lines
corpus = data.lower().split("\n")
# Fit tokenizer on the corpus
tokenizer.fit_on_texts(corpus)
# Get total number of unique words
total_words = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(total_words)

input_sequences = []
for line in corpus:
token_list = tokenizer.texts_to_sequences([line])[0]
for i in range(1, len(token_list)):
n_gram_sequence = token_list[:i+1]
input_sequences.append(n_gram_sequence)
max_sequence_len=max([len(x) for x in input_sequences])
input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len
,padding='pre'))
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
print(xs[5])
model=Sequential()
model.add(Embedding(total_words,32,input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20,return_sequences=True)))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dense(total_words,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accur
acy'])
history=model.fit(xs,ys,epochs=150,verbose=1)
import pickle
# Save the tokenizer to a file
with open('tokenizer.pkl', 'wb') as file:
pickle.dump(tokenizer, file)
# Replace '/path/to/your/directory/' with the actual path on your PC where you
want to save the model
model.save('C:\AI_lyrics/my_model.keras')
model.save("my_model.h5")
seed_text="I walk"
next_words=100
for _ in range(next_words):
token_list=tokenizer.texts_to_sequences([seed_text])[0]

token_list=pad_sequences([token_list],maxlen=max_sequence_len-
1,padding='pre')

predicted=np.argmax(model.predict(token_list,verbose=0),axis=1)
output_word=""
for word,index in tokenizer.word_index.items():
if index==predicted:
output_word=word
break
seed_text += " "+output_word
print(seed_text)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Load the saved model
model_path = 'my_model.h5'
model = load_model(model_path)
# Load the tokenizer
tokenizer_path = 'tokenizer.pkl'
with open(tokenizer_path, 'rb') as handle:
tokenizer = pickle.load(handle)
# Define a seed text (optional)
seed_text = "Her name is Sruti"
# Define the number of words to generate
num_words_to_generate = 100
# Generate lyrics
generated_lyrics = seed_text
for _ in range(num_words_to_generate):
# Tokenize the seed text
token_list = tokenizer.texts_to_sequences([seed_text])[0]
# Pad the sequence to match the expected input shape
token_list = pad_sequences([token_list], maxlen=model.input_shape[1],
padding='pre')
# Predict the next word
predicted_probs = model.predict(token_list, verbose=0)[0]
# Get the index of the word with the highest probability
predicted_index = predicted_probs.argmax()
# Map the index to the corresponding word
output_word = ""
for word, index in tokenizer.word_index.items():
if index == predicted_index:
output_word = word
break
# Update the seed text with the predicted word
seed_text += " " + output_word
generated_lyrics += " " + output_word
# Print generated lyrics
print("Generated Lyrics:")
print(generated_lyrics)
import matplotlib.pylot as plt
#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.title('Mdoel Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['accuracy'],loc='upper left')
plt.show()


