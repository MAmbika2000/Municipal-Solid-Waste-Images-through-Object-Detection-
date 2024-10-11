import warnings
from Evaluation import evaluation
warnings.filterwarnings("ignore")
from ResidualAttentionNetwork import ResidualAttentionNetwork
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np


def Model_PROPOSED(train_data, train_target, test_data, test_target):
    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32
    IMAGE_CHANNELS = 3
    IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    X_test = np.zeros((test_data.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), dtype=np.uint8)
    X_train = np.zeros((train_data.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), dtype=np.uint8)

    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMAGE_WIDTH * IMAGE_HEIGHT, 3))
        X_train[i] = np.reshape(temp, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    for i in range(test_data.shape[0]):
        temp = np.resize(train_data[i], (IMAGE_WIDTH * IMAGE_HEIGHT, 3))
        X_test[i] = np.reshape(temp, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    batch_size = 64

    num_classes = train_target.shape[1]

    STEP_SIZE_TRAIN = len(train_data) // batch_size

    model_path = "/pylon5/cc5614p/deopha32/Saved_Models/cvd-model.h5"

    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True)
    csv_logger = CSVLogger("/pylon5/cc5614p/deopha32/Saved_Models/cvd-model-history.csv", append=True)

    callbacks = [checkpoint, csv_logger]

    # Model Training
    with tf.device('/gpu:0'):
        model = ResidualAttentionNetwork(
            input_shape=IMAGE_SHAPE,
            n_classes=num_classes,
            activation='softmax').build_model()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        try:
            history = model.fit(X_train, steps_per_epoch=STEP_SIZE_TRAIN, verbose=0, callbacks=callbacks, batch_size=256,
                                epochs=150, use_multiprocessing=True, workers=40)
        except:
            pass
        score = model.predict(X_test)
        score[score >= 0.5] = 1
        score[score < 0.5] = 0
        Eval = evaluation(score, test_target)
    return Eval
