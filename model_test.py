from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


model = load_model('model.h5')


def main():
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory='data/test', target_size=(224, 224), classes=['batman', 'superman'], batch_size=10, shuffle=False)

    assert test_batches.n == 16
    assert test_batches.num_classes == 2

    predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

    # Using a confusion matrix to see results better
    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    print(cm)


if __name__ == '__main__':
    main()
