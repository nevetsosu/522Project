# Auto encoder
import tensorflow as tf
from keras import layers, models
import os

print("[AUTOENCODER] Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class AutoEncoder:
    AE_PATH = 'autoencoder.weights.h5'
    E_PATH = 'encoder.weights.h5'

    def __init__(self, input_dim: int, new_dim: int):
        self.autoencoder, self.encoder = self.build_autoencoder(input_dim, new_dim)

    def build_autoencoder(self, input_dim, encoding_dim):
        # define layers
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

        # Create the encoders
        autoencoder = models.Model(input_layer, decoded)
        encoder = models.Model(input_layer, encoded)

        # Compile the autoencoder model
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        return autoencoder, encoder

    def save(self):
        self.autoencoder.save_weights(self.AE_PATH)
        self.encoder.save_weights(self.E_PATH)

    def load(self):
        self.autoencoder.load_weights(self.AE_PATH)
        self.encoder.load_weights(self.E_PATH)

    def fit_transform(self, X, save=False, force_refit=False, **kwargs):
        if os.path.exists(self.AE_PATH) and os.path.exists(self.E_PATH):
            print("[AUTOENCODER] Using loaded weights")
            self.load()
            if not force_refit:
                return self.encoder.predict(X)
            else:
                print("[AUTOENCODER] Refitting loaded weights")
        else:
            print("[AUTOENCODER] Fitting new weights")

        # fit and/or save weights
        self.autoencoder.fit(X, X, **kwargs)
        if save:
            print("[AUTOENCODER] Saving weights")
            self.save()

        return self.encoder.predict(X)