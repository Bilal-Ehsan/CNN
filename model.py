import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class_list = ['batman', 'hulk', 'iron man', 'spiderman', 'superman']


def main():
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory='data/train', target_size=(224, 224), classes=class_list, batch_size=10)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory='data/valid', target_size=(224, 224), classes=class_list, batch_size=10)

    assert train_batches.n == 100
    assert valid_batches.n == 40
    assert train_batches.num_classes == valid_batches.num_classes == 5

    vgg16_model = tf.keras.applications.vgg16.VGG16()

    # Convert Functional model to Sequential model (excluding output layer)
    model = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)

    model.add(Dense(units=5, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x=train_batches,
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        validation_steps=len(valid_batches),
        epochs=15,
        verbose=2
    )

    model.save('model.h5')


if __name__ == '__main__':
    main()
