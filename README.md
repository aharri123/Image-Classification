# Image-Classification

## Business Goal ##
We'll be working with Palomar Medical Center, located in Escondido, California. Initially using a deep neural network, then moving onto a Convolutional Neural Network, we build a model that takes patient X-rays and identifies if the patient has Pneumonia or not. Creating a successful model will help the hospital doctors make a quicker diagnosis, by reducing (if not eliminating totally) the time spent on consulting X-rays before meeting with patients. Once proven successful, It may also help to generalize over to other diseases and ailments that are able to be diagnosed via x-ray, such as broken bones and certain types of cancer.

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

**Our initial deep neural network had 0 hidden layers:**

![initial model](https://user-images.githubusercontent.com/45251340/216473473-85788f0c-9750-46bd-a035-fea52a52af89.JPG)

**Our epochs:**

![5 epochs](https://user-images.githubusercontent.com/45251340/216476152-8bdc6bb8-7dd7-46a4-a738-63d5f5e154e4.JPG)


**Our training and testing accuracy:** 

![initial model accuracy](https://user-images.githubusercontent.com/45251340/216473891-798569a9-3e76-47ce-994e-4665ce48bc9e.JPG)

**Our accuracy and loss curves:**

![initial model curves](https://user-images.githubusercontent.com/45251340/216474012-57f0bb2e-ed3f-47f1-8069-b733e76aff8f.JPG)

**Looks like our model is overfitting, since our training accuracy is higher than our testing accuracy. This is not surprising for an initial model. We can also see that there's no loss occuring for our trainin model, which suggests our model needs some work. Let's add some more layers and see how that affects the model.**

## Deep Neural Network With Hidden Layers ##

![deep model with more layers](https://user-images.githubusercontent.com/45251340/216474359-aac42bd8-7a18-4692-a7c8-1578aff0ea27.JPG)

**Our epochs:**

![5 epochs v2](https://user-images.githubusercontent.com/45251340/216476237-d5a5c55c-dc7a-4ef7-ae72-3c2e964c48e7.JPG)


**Our training and testing accuracy:** 

![deep model with more layers accuracy](https://user-images.githubusercontent.com/45251340/216474474-025b61b8-cfec-4aa0-8f7c-f16e26a85444.JPG)


**Our accuracy and loss curves:**

![model with more layers curves](https://user-images.githubusercontent.com/45251340/216476418-60be30ed-0154-427d-b2b3-175c4f56e6d7.JPG)


**This time our accuracy fell greatly. There's also no loss occuring in our training set still. Let's add some more epochs to see if that has any improvement on our results.**

## Deep Neural Network With More Epochs ##

**We ran the same model as before, except this time we increased the number of epochs from 5 to 10.**

**Our epochs:**

![10 epochs](https://user-images.githubusercontent.com/45251340/216476533-89599e1e-cd1d-42fe-b6d3-78b3f0c99053.JPG)

**Our training and testing accuracy:** 

![model with more epochs accuracy](https://user-images.githubusercontent.com/45251340/216476779-f75ce36c-ae7f-4c77-ae41-2a46ef372417.JPG)

**Our accuracy and loss curves:**

![model with more epochs curve](https://user-images.githubusercontent.com/45251340/216476800-b4906850-eb30-43e6-8a0b-20f48fc8f0be.JPG)

**Our testing accuracy is still lower than our training accuracy, so some overfitting is still going on. However we're seeing some loss occuring in our training model, which is an improvement compared to before. Slightly better results, but let's add some dropout regularization to see if that helps decrease overfitting.**

## Deep Neural Network With Dropout Regularization ##

**Implementing dropout regularization:**

![dropout regularization](https://user-images.githubusercontent.com/45251340/216477420-455c8b70-07db-4b9e-b29e-41560de9453e.JPG)

**Our epochs:**

![dropout reg accuracy](https://user-images.githubusercontent.com/45251340/216477590-f6e8189d-fa05-4d5b-a684-08bfe5cb8f19.JPG)

**Our training and testing accuracy:** 

![dropout reg acc](https://user-images.githubusercontent.com/45251340/216477621-311af745-fcc4-48d2-aff2-1479c7ae0381.JPG)

**Our accuracy and loss curves:**

![dropout reg curves](https://user-images.githubusercontent.com/45251340/216477641-f7f50ce5-d8f1-460e-83c0-2db76a4f2401.JPG)


**That doesn't seem to have helped, and if anything has made everything worse. Let's try a different approach and try using a CNN.**

## Our Initial CNN Model ##

**Our initial CNN model had 2 convolutional layers, 3 pooling layers, 1 flattening layer and 1 fully connected dense layer:**

![initial cnn](https://user-images.githubusercontent.com/45251340/216479147-4416d5f0-437f-4fd6-acaa-bd90a17e981d.JPG)

![initial cnn2](https://user-images.githubusercontent.com/45251340/216479154-8f8e741d-f926-45b7-975e-28e564b7b644.JPG)


**Our initial CNN had 20 epochs:**

![initial cnn epochs](https://user-images.githubusercontent.com/45251340/216479556-1dc56de8-8492-402c-9903-2b1db7f9d679.JPG)

**Our training and testing accuracy:** 

![initial cnn accuracy](https://user-images.githubusercontent.com/45251340/216479758-5d6c5e6f-9bb6-4550-b666-1587a0b0d61f.JPG)

**Our accuracy and loss curves:**

![intial cnn curves](https://user-images.githubusercontent.com/45251340/216479789-749a7875-229f-4f1e-89b5-256135ff8ae5.JPG)

**Our training accuracy is higher than our testing accuracy, signifying overfitting but significantly less so when compared to our DNN models. We also immediately see training loss occuring in our initial model, which we did not see in our initial DNN model. Our initial results look promising, so let's try adding some dropout regularization to see if we can improve our model.**

## CNN Model With Dropout Regularization ##

**Implementing dropout regularization:**

![cnn dropout reg](https://user-images.githubusercontent.com/45251340/216480307-a82831ed-5ba5-4eef-9cbc-1cd20c5fa8d1.JPG)

**Our epochs:**

![cnn dropout reg epochs](https://user-images.githubusercontent.com/45251340/216480524-9a3fefb3-dc26-4d18-8496-82d95cfde6e9.JPG)

**Our training and testing accuracy:** 

![cnn dropout reg accuracy](https://user-images.githubusercontent.com/45251340/216480558-ad516727-31a4-4d60-82a1-b93026ed6536.JPG)

**Our accuracy and loss curves:**

![cnn dropout reg curves](https://user-images.githubusercontent.com/45251340/216480570-876857f1-75c2-4ec8-81ff-2c3678de961b.JPG)

**Our training and testing accuracy are now the same, which is an improvement. Our training and testing loss are also extremely close in value. However our validation accuracy is still relatively low. Let's see what happens if we decrease the number of epochs to around 10.**

## CNN Model With Decreased Epochs##
We used the same model as previous, but changed our number of epochs from 20 to 10.

**Our epochs:**

![cnn decrease epochs epochs](https://user-images.githubusercontent.com/45251340/216481212-3dd92522-b694-477a-870f-ea8c71722462.JPG)

**Our training and testing accuracy:** 

![cnn decrease epochs accuracy](https://user-images.githubusercontent.com/45251340/216481252-8fd529d4-a379-49d5-8fcc-1d229a0b9514.JPG)

**Our accuracy and loss curves:**

![cnn decrease epochs curves](https://user-images.githubusercontent.com/45251340/216481262-5cb70d4e-c757-4cb4-bd80-6ad0bcbed319.JPG)

**For the first time, our testing accuracy is higher than our training accuracy. Our validation accuracy is also the highest it has been during our tuning. This is significant improvement from our initial DNN and CNN model, so let's move on to our conclusion.**

## Conclusion ##

Our goal was to work with Palomar Medical Center to build an image classification model that takes patient X-rays and identifies if the patient has Pneumonia or not. This will help the hospital doctors make a quicker diagnosis, and reduce the time spent on consulting X-rays before meeting with patients. We initially used a deep neural network, then moved onto a Convolutional Neural Network.

Our initial deep neural network had 0 hidden layers, and after running we found it was overfitting. Our model had a testing accuracy of about 92% and a testing accuracy of 72%. For the next model, we added more layers, and after running it found that while accuracy dramatically dropped, it was still overfitting. It had a training accuracy of about 39% and a testing accuracy of about 34%. Next, we added more epochs to see if that would have an impact on our accuracy and overfitting issue. We found that we still had an overfitting issue, but our accuracy improved again, with our training accuracy at about 94%, and our testing accuracy at 79%. Lastly, we implemented dropout regularization, to see if that would reduce overfitting. Our training accuracy was 74%, while our testing accuracy was about 62%, so overfitting was still an issue. We then decided to switch over to a CNN.

Our initial CNN model had 2 convolutional layers, 3 pooling layers, 1 flattening layer and 1 fully connected dense layer. Our initial training accuracy was about 95% and our testing accuracy was about 94%. This along with our loss and accuracy curves sugested overfitting still. Next we implemented dropout regularization before doing anything else to see if our overfitting would be addressed. Our curves improved slightly, and our testing and training accuracy were both at about 94%. Lastly, we decreased the number of epochs from 20 to 10. Here, we saw the most improvement, not only in our curves but also in our accuracy. Our final training accuracy was about 90%, and our final testing accuracy was about 91%.

Our model accuracy overall improved greatly when switching from a deep neural network to a CNN. At it's lowest, our deep neural network had a testing accuracy of about 34%, and at its highest it was about 79%. In comparison, our CNN had at its lowest a testing accuracy of about 91%, and at its highest 94%. Our model is still overfitting, but compared to previous models, we were able to reduce it somewhat through Dropout Regularization and reducing the number of epochs.
