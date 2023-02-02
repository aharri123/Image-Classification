# Image-Classification

## Business Goal ##
We'll be working with Palomar Medical Center, located in Escondido, California. Initially using a deep neural network, then moving onto a Convolutional Neural Network, we build a model that takes patient X-rays and identifies if the patient has Pneumonia or not. Creating a successful model will help the hospital doctors make a quicker diagnosis, and reduce the time spent on consulting X-rays before meeting with patients.

## Data ##
The data comes from a kaggle dataset of chest X-ray images. There are 5,856 images in total, and the data is split into train, test and validation folders. Each folder is further split into folders that contain Pneumonia and non Pneumonia X-rays. The images are split up in the folders as follows: 


![training images](https://user-images.githubusercontent.com/45251340/216470311-bab2ac1f-fd4d-46d2-be38-2fc4ad43e7a9.JPG)


![testing images](https://user-images.githubusercontent.com/45251340/216470408-b3674dbe-f612-4778-8136-04066fbe1a1e.JPG)


![validation images](https://user-images.githubusercontent.com/45251340/216470461-c531972d-4403-4f35-a633-4d6d49dd1b42.JPG)


## Starting off ##
First we start off by taking all the data from our directories and augmenting them: 

![reshaping](https://user-images.githubusercontent.com/45251340/216470702-8404dbad-3b62-4bfb-83ca-a07a2b594390.JPG)

After augmenting, we'll create our training, testing, and validation sets, then reshape our images.  
