# plate_detection
cool project to learn more about unsupervised image processing

I found a dataset of images from one of the Customs in Iran that contains images of trucks, maybe wants to come in or exit this country.

the objective is finding plates of these trucks and then recognize the plate numbers. in most of the images, trucks have two plates, an Iranian plate, and an International plate.

here is one example:

![2510.jpg](https://github.com/MSadeghzadehG/plate_detection/blob/master/2510.jpg "Example image")

first I tried to find plates. so I did some preprocessing on the image: rotation and binarization.

![plot1.png](https://github.com/MSadeghzadehG/plate_detection/blob/master/plot1.png "preprocessed image")

then I found contours in the image and according to the size of plates in the dataset images, select the contours which probably is a plate.

![plot2.png](https://github.com/MSadeghzadehG/plate_detection/blob/master/plot2.png "plates image")

now we need to do segmentation on the plates to find characters. like the last part, I found the contours and then select the characters.

in the last step, we need to recognize the characters. I have used some trained model but they don't work. so I trained a model using the mnist dataset to recognize the International plate numbers.

my model appeared better and I used that. at the end of my processing, my code creates an image which named with predicted numbers and shows the segmentation of the plate.

![prediction.png](https://github.com/MSadeghzadehG/plate_detection/blob/master/prediction.png "prediction image")

## to-do
* find a better way to detect plates
* recognize Persian characters
