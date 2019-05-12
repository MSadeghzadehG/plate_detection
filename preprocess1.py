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
gray_car_image = car_image * 255
rotated_image = rotate_image(gray_car_image)
print(car_image.shape)

# the next line is not compulsory however, a grey scale pixel
# in skimage ranges between 0 & 1. multiplying it with 255
# will make it range between 0 & 255 (something we can relate better with


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(binarization(rotated_image), cmap="gray")
ax2.imshow(, cmap="gray")
plt.show()

