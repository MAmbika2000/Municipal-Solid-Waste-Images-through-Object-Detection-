from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from Evaluation import evaluation
import cv2 as cv
import numpy as np


def Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target):
    # Define the input shape
    IMG_SIZE = (255, 255, 3)  # Adjust the shape based on your data
    num_classes = Test_Target.shape[-1]

    Feat1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat1[i, :] = np.resize(Train_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    train_data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Feat2[i, :] = np.resize(Test_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])


    # Load the InceptionV3 model (pre-trained on ImageNet)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  #
    predictions = Dense(num_classes, activation='softmax')(x)  # 'softmax'
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Fit the model
    model.fit(train_data, Train_Target, epochs=150, batch_size=256, steps_per_epoch=1000, validation_data=(test_data, Test_Target))
    pred = model.predict(test_data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Test_Target)
    return Eval