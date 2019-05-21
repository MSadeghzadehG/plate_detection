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



def rotate_image(inital_image):
    return rotate(inital_image, -11, center=None)


def binarization(input_image):
    threshold_value = threshold_otsu(input_image)
    return input_image > threshold_value


car_image = imread("2454.jpg", as_grey=True)
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
plate_dimensions = (35, 60, 160, 215)
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


import pickle
print("Loading model")
filename = './finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

for plate in plate_like_objects:
    characters = []
    binary_plate = binarization(plate)
    license_plate = np.invert(binary_plate)

    cv_image = img_as_ubyte(license_plate)
    img = cv_image

    # find contours
    ctrs, hier = cv2.findContours(cv_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = img[y:y+h, x:x+w]

        # show ROI
        # cv2.imwrite('roi_imgs.png', roi)
        characters.append(cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA))
        # converts it to a 1D array
        each_character = characters[-1].reshape(1, -1)
        result = model.predict(each_character)
        print(result[0])
        # classification_result.append(result)
        cv2.imshow('charachter'+str(i), characters[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)

    print('found')
    cv2.imshow('marked areas', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    

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
