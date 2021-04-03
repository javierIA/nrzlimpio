import time
import tensorflow as tf
import cv2
import numpy as np
from skimage import measure  ##Calcule el índice de similitud estructural medio entre dos imágenes.
import cvsdata
import argparse



def show_window():
    num = 0

    for (x, y, window) in sliding_window(resized, 5, (wind_row, wind_col)):
        if window.shape[0] != wind_row or window.shape[1] != wind_col:
            continue
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + wind_row, y + wind_col), (0, 500, 0), 2)

        t_img = resized[y:y + wind_row, x:x + wind_col]  # the image which has to be predicted
        # expanding the dimensions of the image to meet the dimensions of the trained model

        ##input_data =np.array(test_img, dtype=np.float32)
        # interpreter.set_tensor(input_details[0]['index'], input_data)

        # interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        # output_data = interpreter.get_tensor(output_details[0]['index'])

        classes = loadmodel(t_img)

        if classes[1] > 0.99999:  # Applying threshold
            print("Vegetation detected with a probability: ", classes[0], '\t', "x: ", x, '\t', "y: ", y)
            num += 1
            savedata.append([x, y, num])
            S = compare_images(t_img, tp)
            scp.append(S)
            if S > 0.5:
                TP.append(1)
            else:
                FP.append(1)

        else:
            print("background with a probability: ", 1 - classes[1], '\t', "x: ", x, '\t', "y: ", y)
            S = compare_images(t_img, tn)
            scn.append(S)
            if S > 0.5:
                TN.append(1)
            else:
                FN.append(1)

        cv2.imshow("sliding_window", resized[y:y + wind_row, x:x + wind_col])
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.25)


def compare_images(imageA, imageB):
    return measure.compare_ssim(imageA, imageB, multichannel=True)


# generating the sliding window
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def consfusion_matrix(TN, FN, TP, FP):
    print("The recall of the model: ", len(TP) / (len(TP) + len(FN)))
    print("The F1 score of the model", (2 * len(TP)) / (len(FP) + len(FN) + 2 * len(TP)))


def loadmodel(ImageT):


    img_rgb = ImageT
    #convert img to input tensor
    img_rgb = cv2.resize(img_rgb, (180, 180), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 180, 180, 3])
    img_rgb = np.array(img_rgb, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], img_rgb)
    interpreter.invoke()
    box = interpreter.get_tensor(output_details[0]['index'])[0]
    return box


if __name__ == '__main__':
    savedata = []
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image")
    ap.add_argument("-n", "--name", required=True,
                    help="name of the user")
    args = vars(ap.parse_args())
    # load the input image from disk
    img = cv2.imread(args["input"])

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    TP =[]
    TN =[]
    FP =[]
    FN =[]
    scp = []
    scn = []

    ##Arrays for the skimage library
    #TP=TRUE POSITVE
    #TN=TRUE NEGATIVE
    #FN=FALSE NEGATIVE
    #FN=FALSE NEGATVE
    true_positive = cv2.imread('images/class2/AV1.png')
    true_negative = cv2.imread('images/class1/AC1.png')
    tp = true_positive[:,:]/img.shape[0]
    tn = true_negative[:,:]/img.shape[1]
    resized = img# Pre-processing the image and normalize
    wind_row, wind_col = 20,20 # dimensions of the image
    img_rows, img_cols = 20,20
    show_window()
    consfusion_matrix(TN,FN,TP,FP)
    cvsdata.datacsv(savedata,args["name"])
    print('\t',"TN=",len(TN),'\t',"FP=",len(FP))
    print('\t',"FN=",len(FN),'\t',"TP=",len(TP))
    cv2.waitKey(0)
    cv2.destroyAllWindows()