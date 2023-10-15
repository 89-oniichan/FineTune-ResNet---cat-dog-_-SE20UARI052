import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow_datasets as tfds

# Load the dataset
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Resize images to the expected ResNet input size
IMG_SIZE = 224
def format_example(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Shuffle and batch the data
BATCH_SIZE = 32
train_batches = train.shuffle(1000).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation.batch(BATCH_SIZE).prefetch(1)
test_batches = test.batch(BATCH_SIZE)

# Create the base model from pre-trained ResNet50
base_model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                      include_top=False,
                      weights='imagenet')

# Unfreeze the last few layers of the base model for fine-tuning
base_model.trainable = True

# Define the layers to be fine-tuned
fine_tune_at = 100  # Adjust this number based on the architecture of ResNet50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Add custom layers for classification
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Adjust the learning rate as needed
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model, including fine-tuning
initial_epochs = 5
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history = model.fit(train_batches,
                    epochs=total_epochs,
                    validation_data=validation_batches,
                    initial_epoch=0)

# Evaluate the model
loss, accuracy = model.evaluate(test_batches)
print("\nTest accuracy: {:.2f}%".format(accuracy * 100))
