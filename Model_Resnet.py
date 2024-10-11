import numpy as np
import cv2 as cv
from keras import Sequential, Model
from keras.applications import ResNet50
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from Evaluation import evaluation


def Model_RESNET(Train_Data, train_target, Test_Data, Test_Target):
    IMG_SIZE = (32, 32, 3)
    num_classes = Test_Target.shape[-1]
    Feat1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat1[i, :] = np.resize(Train_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    train_data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Feat2[i, :] = np.resize(Test_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    # Load the ResNet50 model with pre-trained ImageNet weights, excluding the top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(train_target.shape[1], activation='softmax')(x)
    # Define the new model
    model = Model(inputs=base_model.input, outputs=x)
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(train_data, train_target, epochs=150, batch_size=255, validation_split=0.1)
    pred = model.predict(test_data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Test_Target)
    return Eval


