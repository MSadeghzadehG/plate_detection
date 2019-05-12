from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import cv2


def rotate_image(inital_image):
    return rotate(inital_image, -11, center=None)


def binarization(rotated_image):
    threshold_value = threshold_otsu(rotated_image)
    return rotated_image > threshold_value



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
    # minRow, minCol, maxRow, maxCol = region.bbox
    # rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
    # ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions

    
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    # print(region_height,region_width)
    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        print(region_height,region_width)
        plate_like_objects.append(gray_car_image[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                              max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions

plt.show()

