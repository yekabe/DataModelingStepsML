#%%
# from notebooks.deep_learning.cifar10 import x_train
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

print(X_train.shape)
print(X_test.shape)

# %%
input_img= Input(shape=(784,))


# %%
encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)
encoded = Dense(units=32, activation='relu')(encoded)
encoded = Dense(units=16, activation='relu')(encoded)
encoded = Dense(units=10, activation='relu')(encoded)
decoded = Dense(units=16, activation='relu')(encoded)
decoded = Dense(units=32, activation='relu')(decoded)
decoded = Dense(units=64, activation='relu')(decoded)
decoded = Dense(units=128, activation='relu')(decoded)
decoded = Dense(units=784, activation='sigmoid')(decoded)

# %%
autoencoder=Model(input_img, decoded)
autoencoder.summary()


# %%
encoder = Model(input_img, encoded)
encoder.summary()

# %%
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
autoencoder.fit(X_train, X_train,
                epochs=40,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# %%
encoded_imgs = encoder.predict(X_test)
predicted = autoencoder.predict(X_test)

# %%
plt.figure(figsize=(40, 4))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display encoded images
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_imgs[i].reshape(5,2))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# display reconstructed images
    ax = plt.subplot(3, 20, 2*20 +i+ 1)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  
    
plt.show()



# %%
from sklearn import svm

X_enc = encoder.predict(X_train)
X_test_enc = encoder.predict(X_test)

print(X_train.shape, X_enc.shape)

#%%

clf = svm.SVC( )
clf.fit(X_enc, y_train)

# %%

clf.score(X_test_enc, y_test)

# %%
