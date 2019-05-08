#!/usr/bin/env python
# coding: utf-8

# In[63]:


from keras.datasets import imdb
(train_data, train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)


# In[64]:


len(train_data)


# In[65]:


len(test_data)


# In[66]:


train_data[10]


# In[67]:


train_labels[0]


# In[68]:


max([max(sequence) for sequence in train_data])


# In[69]:


word_index = imdb.get_word_index()
reverse_word_index=dict([(value, key) for (key,value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])


# In[70]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[71]:


x_train[0]


# In[72]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[73]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[74]:


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


# In[75]:


from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])


# In[76]:


from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])


# In[77]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[78]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy',  metrics=['acc'])

history = model.fit(partial_x_train,partial_y_train, epochs=25,batch_size=512, validation_data=(x_val, y_val))


# In[79]:


history_dict = history.history
history_dict.keys()


# In[80]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_loss_values = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[81]:


plt.clf()
acc_values = history_dict['acc'] 
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[82]:


model = models.Sequential()
model.add(layers.Dense(16, activation='sigmoid', input_shape=(10000,)))
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.Dense(1, activation='relu'))


model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
model.fit(x_train, y_train, epochs=25, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[31]:


results


# In[32]:


model.predict(x_test)


# In[34]:


from keras.preprocessing.text import text_to_word_sequence
text_file = open("Input.txt","r")
lines = text_file.readlines()
print (lines)


# In[35]:


file = []
for c in lines:
    
    words = c.split(" ")
    print(words)
    #words = lines.split()
    #for words in lines:
        #file.append(words)
file = words


# In[36]:


file_size = len(file)
print(file_size)
type(file)
file=''.join(file)
type(file)


# In[37]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
t=Tokenizer()
t.fit_on_texts(file)

input = hashing_trick(file,500, hash_function = 'md5')


# In[38]:


encoded_docs =  t.texts_to_matrix(file,mode = 'count')
print(encoded_docs)


# In[39]:


word_index = imdb.get_word_index()


# In[40]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_test = vectorize_sequences(input)


# In[41]:


model.predict(x_test)


# In[42]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_loss_values = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




