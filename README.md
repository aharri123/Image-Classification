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

**Our initial deep neural network had 0 hidden layers: **

![initial model](https://user-images.githubusercontent.com/45251340/216473473-85788f0c-9750-46bd-a035-fea52a52af89.JPG)

**Our epochs:**

![5 epochs](https://user-images.githubusercontent.com/45251340/216476152-8bdc6bb8-7dd7-46a4-a738-63d5f5e154e4.JPG)


**Our training and testing accuracy:** 

![initial model accuracy](https://user-images.githubusercontent.com/45251340/216473891-798569a9-3e76-47ce-994e-4665ce48bc9e.JPG)

**Our accuracy and loss curves:**

![initial model curves](https://user-images.githubusercontent.com/45251340/216474012-57f0bb2e-ed3f-47f1-8069-b733e76aff8f.JPG)

As we can see, our model is overfitting. So for our next model, we added some more layers.

## Deep Neural Network With Hidden Layers ##

![deep model with more layers](https://user-images.githubusercontent.com/45251340/216474359-aac42bd8-7a18-4692-a7c8-1578aff0ea27.JPG)

**Our epochs:**

![5 epochs v2](https://user-images.githubusercontent.com/45251340/216476237-d5a5c55c-dc7a-4ef7-ae72-3c2e964c48e7.JPG)


**Our training and testing accuracy:** 

![deep model with more layers accuracy](https://user-images.githubusercontent.com/45251340/216474474-025b61b8-cfec-4aa0-8f7c-f16e26a85444.JPG)


**Our accuracy and loss curves:**

![model with more layers curves](https://user-images.githubusercontent.com/45251340/216476418-60be30ed-0154-427d-b2b3-175c4f56e6d7.JPG)


From this, we can see that our accuracy dramatically dropped, but that we're still overfitting. Next, we'll try adding more epochs to see if that has an effect.

## Deep Neural Network With More Epochs ##

We ran the same model as before, except this time we increased the number of epochs from 5 to 10.

**Our epochs:**

![10 epochs](https://user-images.githubusercontent.com/45251340/216476533-89599e1e-cd1d-42fe-b6d3-78b3f0c99053.JPG)

**Our training and testing accuracy:** 

![model with more epochs accuracy](https://user-images.githubusercontent.com/45251340/216476779-f75ce36c-ae7f-4c77-ae41-2a46ef372417.JPG)

**Our accuracy and loss curves:**

![model with more epochs curve](https://user-images.githubusercontent.com/45251340/216476800-b4906850-eb30-43e6-8a0b-20f48fc8f0be.JPG)

We still have an overfitting issue, but our accuracy is back up again. Lastly, let's see if we can implement dropout regularization to reduce overfitting.

## Deep Neural Network With Dropout Regularization ##

**Implementing dropout regularization:**

![dropout regularization](https://user-images.githubusercontent.com/45251340/216477420-455c8b70-07db-4b9e-b29e-41560de9453e.JPG)

**Our epochs:**

![dropout reg accuracy](https://user-images.githubusercontent.com/45251340/216477590-f6e8189d-fa05-4d5b-a684-08bfe5cb8f19.JPG)

**Our training and testing accuracy:** 

![dropout reg acc](https://user-images.githubusercontent.com/45251340/216477621-311af745-fcc4-48d2-aff2-1479c7ae0381.JPG)

**Our accuracy and loss curves:**

![dropout reg curves](https://user-images.githubusercontent.com/45251340/216477641-f7f50ce5-d8f1-460e-83c0-2db76a4f2401.JPG)


From our accuracy scores and curves, we can see that overfitting is still an issue. So let's try and switch over to a CNN model, and see if we can improve our output.

## Our Initial CNN Model ##

**Our initial CNN model has 2 convolutional layers, 3 pooling layers, 1 flattening layer and 1 fully connected dense layer:**

![initial cnn](https://user-images.githubusercontent.com/45251340/216479147-4416d5f0-437f-4fd6-acaa-bd90a17e981d.JPG)

![initial cnn2](https://user-images.githubusercontent.com/45251340/216479154-8f8e741d-f926-45b7-975e-28e564b7b644.JPG)



