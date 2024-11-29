
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load dataset - use a pre-trained medical image dataset for simplicity
# For example, the Kaggle Chest X-ray dataset
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()  # Replace this with your medical dataset

# Normalize the images to [-1, 1]
x_train = x_train.astype('float32') / 255.0
x_train = (x_train - 0.5) * 2.0  # Normalize to [-1, 1]

# Create the Generator model
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))
    model.add(layers.Reshape((8, 8, 128)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.ReLU())
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.ReLU())
    model.add(layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh"))
    return model

# Create the Discriminator model
def build_discriminator(image_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=image_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

# Compile the models
latent_dim = 100
image_shape = (32, 32, 3)  # Modify for medical images size (e.g., 64x64 for medical images)

# Create the generator and discriminator
generator = build_generator(latent_dim)
discriminator = build_discriminator(image_shape)

# Make the discriminator not trainable when training the GAN model
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
discriminator.trainable = False

# GAN model that combines generator and discriminator
gan_input = layers.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)

# Compile GAN model
gan.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training loop for GAN
def train_gan(epochs, batch_size=128):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Train discriminator with real and fake images
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_images = x_train[idx]
        fake_images = generator.predict(np.random.randn(half_batch, latent_dim))
        
        # Labels for real and fake images
        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))
        
        # Train on real and fake images
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator (through the GAN)
        noise = np.random.randn(batch_size, latent_dim)
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)
        
        # Print the progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | Acc.: {100*d_loss[1]}] [G loss: {g_loss[0]} | Acc.: {100*g_loss[1]}]")

        # Save generated images every few epochs
        if epoch % 500 == 0:
            save_generated_images(epoch)

def save_generated_images(epoch, examples=10, latent_dim=100):
    noise = np.random.randn(examples, latent_dim)
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale images to [0, 1]
    
    plt.figure(figsize=(10, 10))
    for i in range(examples):
        plt.subplot(1, examples, i + 1)
        plt.imshow(generated_images[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"generated_images_{epoch}.png")
    plt.close()

# Train the GAN
train_gan(epochs=10000, batch_size=64)
