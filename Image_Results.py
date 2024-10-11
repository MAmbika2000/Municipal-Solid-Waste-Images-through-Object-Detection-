import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
no_of_dataset = 1

def Image_Result():
    for n in range(no_of_dataset):
        Images = np.load('Image.npy', allow_pickle=True)
        seg = np.load('Segmentation.npy', allow_pickle=True)
        Image = [[0, 1, 2, 3, 4]]
        im = [[0, 1, 2, 3, 4]]
        for i in range(len(Image[0])):
            fig, ax = plt.subplots(1, 2)
            plt.suptitle("Image Result", fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title('Original Image', fontsize=10)
            plt.imshow(Images[Image[n][i]])
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title('Segmented Image', fontsize=10)
            plt.imshow(seg[im[n][i]])
            plt.axis('off')
            fig.tight_layout()
            plt.show()

            cv.imwrite('./Results/original-' + str(i + 1) + '.png', Images[Image[n][i]])
            cv.imwrite('./Results/segment-' + str(i + 1) + '.png', seg[im[n][i]])


if __name__ == '__main__':
    Image_Result()
