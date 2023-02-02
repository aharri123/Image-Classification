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

**After augmenting, we'll create our training, testing, and validation sets, then reshape our images. Before reshaping, our image set sizes were:** 

![before resizing data](https://user-images.githubusercontent.com/45251340/216472303-a48413cd-aabb-46dc-a73d-c7fb5a49bec8.JPG)

**After reshaping our images, the sizes are:**

![after resizing data](https://user-images.githubusercontent.com/45251340/216472520-fdaff589-bb7c-47b5-b9b9-5e213bcf2354.JPG)

**Now it's time to build our initial model**

## Initial Model ##

**Our initial deep neural network had 0 hidden layers**

![initial model](https://user-images.githubusercontent.com/45251340/216473473-85788f0c-9750-46bd-a035-fea52a52af89.JPG)

**Our training and testing accuracy:** 

![initial model accuracy](https://user-images.githubusercontent.com/45251340/216473891-798569a9-3e76-47ce-994e-4665ce48bc9e.JPG)

**Our accuracy and loss curves:**

![initial model curves](https://user-images.githubusercontent.com/45251340/216474012-57f0bb2e-ed3f-47f1-8069-b733e76aff8f.JPG)

As we can see, our model is overfitting. So for our next model, we added some more layers.

## Deep neural network with hidden layers ##

![deep model with more layers](https://user-images.githubusercontent.com/45251340/216474359-aac42bd8-7a18-4692-a7c8-1578aff0ea27.JPG)

**Our training and testing accuracy:** 

![deep model with more layers accuracy](https://user-images.githubusercontent.com/45251340/216474474-025b61b8-cfec-4aa0-8f7c-f16e26a85444.JPG)

