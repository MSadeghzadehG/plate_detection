from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import numpy as np
from skimage.transform import resize
import cv2
from skimage import img_as_ubyte

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def rotate_image(inital_image):
    return rotate(inital_image, -11, center=None)


def binarization(input_image):
    threshold_value = threshold_otsu(input_image)
    return input_image > threshold_value


car_image = imread("2510.jpg", as_grey=True)
gray_car_image = rotate_image(car_image * 255)
binary_car_image = binarization(gray_car_image)
print(car_image.shape)

# the next line is not compulsory however, a grey scale pixel
# in skimage ranges between 0 & 1. multiplying it with 255
# will make it range between 0 & 255 (something we can relate better with

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(binary_car_image, cmap="gray")
ax2.imshow(gray_car_image, cmap="gray")
plt.show()

# this gets all the connected regions and groups them together
label_image = measure.label(binary_car_image)
plate_dimensions = (35, 60, 140, 215)
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []
fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")
# contours, hierarchy = cv2.findContours(gray_car_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue

    # the bounding box coordinates
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="green", linewidth=1, fill=False)
    ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions

    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    # print(region_height,region_width)
    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        print(region_height, region_width)
        plate_like_objects.append(gray_car_image[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions

plt.show()


# import pickle
# print("Loading model")
# filename = './finalized_model.sav'
# model = pickle.load(open(filename, 'rb'))

for plate in plate_like_objects:
    prediction = ''
    characters = []
    binary_plate = binarization(plate)
    license_plate = np.invert(binary_plate)
    # license_plate = binary_plate

    cv_image = img_as_ubyte(license_plate)
    img = cv_image
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    # find contours
    ctrs, hier = cv2.findContours(cv_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        if w > 40 or h > 40 or h < 20:
            continue
        # Getting ROI
        roi = img[max(y-5, 0):y+h+5, max(0, x-5):x+w+5,0]

        # show ROI
        characters.append(cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA))
        
        # usning TFKpredict
        # x = characters[-1].reshape(1, 784).astype('float32') / 255
        # from keras.models import load_model
        # model = load_model('TFKeras.h5')
        # out = model.predict(x)
        # print(np.argmax(out))
        
        # using cnnPrecict
        # x = characters[-1].reshape(1,28,28,1)
        # x = x.astype('float32')
        # x /= 255
        # from keras.models import load_model
        # model = load_model('cnn.h5')
        # out = model.predict(x)
        # print(np.argmax(out))
        
        # using my model
        char = characters[-1].reshape(1, 28, 28, 1)
        char = char.astype('float32')
        char /= 255
        from keras.models import load_model
        model = load_model('my_model.h5')
        out = model.predict(char)
        print(out)
        prediction += str(np.argmax(out))

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imshow('charachter'+str(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('found')
    cv2.imwrite(prediction+'.png', img)



    

    # print('Model loaded. Predicting characters of number plate')
    # classification_result = []
    # for each_character in characters:
        

    # print('Classification result')
    # print(classification_result)

    # plate_string = ''
    # for eachPredict in classification_result:
    #     plate_string += eachPredict[0]

    # print('Predicted license plate')
    # print(plate_string)
