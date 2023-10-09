import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds



# shuffle_files=True: good practice to shuffle the data if it's on multiple files
# as_supervised=True: returns a tuple (img, label) instead of a dictionary {'image': img, 'label': label}


mnist = tf.keras.datasets.mnist


(ds_train, ds_test), df_info = tfds.load('mnist', split=['train','test'], shuffle_files=True, as_supervised=True, with_info=True,)

#normalize the images becuase the dataset provides type tf.uint8, but the model expects tf.float32
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

#cache the dataset before shuffling to increase performance
ds_train = ds_train.cache()

#for true randomness, set the shuffle buffer to the full dataset size
ds_train = ds_train.shuffle(df_info.splits['train'].num_examples)

#batch elements of the ds after shuffling to get unique batches at each epoch
ds_train = ds_train.batch(128)

#Good practice to end the pipeline by prefetching for performance (Reduced data loading bottleneck, parallelism(overlap data preprocessing with augmentation and normalization))
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


#fit (train) the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,    
)

model.save('optimizedHandwritten.model')

# we already trained the model, so we can just load it
model = tf.keras.models.load_model('optimizedHandwritten.model')

# loss, accuracy = model.evaluate(ds_train, ds_test)

# print(loss)
# print(accuracy)

image_number = 0

while os.path.isfile(f"TestDigits/Untitled{image_number}.png"):
    try:
        img = cv2.imread(f"TestDigits/Untitled{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1