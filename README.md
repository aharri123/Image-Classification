# Image-Classification

## Business Goal ##
We'll be working with Palomar Medical Center, located in Escondido, California. Initially using a deep neural network, then moving onto a Convolutional Neural Network, we build a model that takes patient X-rays and identifies if the patient has Pneumonia or not. Creating a successful model will help the hospital doctors make a quicker diagnosis by reducing (if not eliminating totally) the time spent on consulting X-rays before meeting with patients. Once proven successful, It may also help to generalize over to other diseases and ailments that are able to be diagnosed via x-ray, such as broken bones and certain types of cancer.

## Data ##
The data comes from a kaggle dataset of chest X-ray images. There are 5,856 images in total, and the data is split into train, test and validation folders. Each folder is further split into folders that contain Pneumonia and non Pneumonia X-rays. The images are split up in the folders as follows: 


![training images](https://user-images.githubusercontent.com/45251340/216470311-bab2ac1f-fd4d-46d2-be38-2fc4ad43e7a9.JPG)


![testing images](https://user-images.githubusercontent.com/45251340/216470408-b3674dbe-f612-4778-8136-04066fbe1a1e.JPG)


![validation images](https://user-images.githubusercontent.com/45251340/216470461-c531972d-4403-4f35-a633-4d6d49dd1b42.JPG)


**From our image distributions, we can see that there is a data imbalance, with there being more Pneumonia x-rays than normal x-rays**

## Starting off ##
First we start off by taking all the data images from our directories and augmenting them into a 200 x 200 pixel size. After that, we create our training, testing and validation data sets. Finally we reshape our images. Our initial sets had the following shapes:

* X_train shape: (5216, 200, 200, 3)
* y_train shape: (5216, 2)
* X_test shape: (624, 200, 200, 3)
* y_test shape: (624, 2)
* X_val shape: (16, 200, 200, 3)
* y_val shape: (16, 2)

**After reshaping our data, we had the following shapes:**

* X_train shape: (5216, 120000)
* y_train shape: (5216, 1)
* X_test shape: (624, 120000)
* y_test shape: (624, 1)
* X_val shape: (16, 120000)
* y_val shape: (16, 1)

**We can now move on to bulding our initial DNN model**

## Initial DNN Model ##

**Our initial deep neural network had 0 hidden layers:**

![initial dnn model summary](https://user-images.githubusercontent.com/45251340/219884458-7f93527f-48ba-4586-8674-fc7b1e618a25.JPG)


**Our accuracy and loss curves:**

![initial model curves](https://user-images.githubusercontent.com/45251340/216474012-57f0bb2e-ed3f-47f1-8069-b733e76aff8f.JPG)

**From our model we can see:**
* Our model had a training accuracy of 92%
* Our model had a testing accuracy of 72% 
* Our model had a validation accuracy of 75%

**Looks like our model is overfitting, since our training accuracy is higher than our testing accuracy. This is not surprising for an initial model. We can also see that there's no loss occuring in our training loss curve, which suggests our model needs some work.**


## Further Attempts To Tune Our DNN Model ##
We tuned our model several more times, trying different things like adding more layers, adding more epochs and utilizing dropout regularization. The results of our different models are summed up below: 

| **Tuning Technique**   	| **Training Accuracy** 	| **Testing Accuracy** 	| **Validation Accuracy** 	|
|------------------------	|-----------------------	|----------------------	|-------------------------	|
| Adding hidden layers   	| 39%                   	| 34%                  	| 81%                     	|
| Adding more epochs     	| 94%                   	| 79%                  	| 79%                     	|
| Dropout regularization 	| 74%                   	| 62%                  	| 62%                     	|

It seems that our tuning methods aren't drastically improving our model. This could be due to our imbalance of data mentioned earlier, and also the type of model (DNN) that we're using. Let's try a different approach and try using a CNN, while also addressing the imbalance of data. We'll also start implementing classification reports (mainly for recall) and confusion matrices into our results.

## Our Initial CNN Model ##

**Our initial CNN model had 3 convolutional layers, 3 pooling layers, 1 flattening layer and 1 fully connected dense layer:**


![initial cnn2](https://user-images.githubusercontent.com/45251340/216479154-8f8e741d-f926-45b7-975e-28e564b7b644.JPG)


**Our accuracy and loss curves:**

![intial cnn curves](https://user-images.githubusercontent.com/45251340/216479789-749a7875-229f-4f1e-89b5-256135ff8ae5.JPG)

**From our model accuracy we can see:**
* Our model had a training accuracy of 96%
* Our model had a testing accuracy of 97% 
* Our model had a validation accuracy of 75%

****Let's take a look at our classification report results:** 
For our test model class 1 (meaning a chest x-ray is classified as Pneumonia) we have a precision score of .97, a recall score of .99, and an f1 score of .98, meaning:

* Out of all the x-rays that the model classified as Pneumonia, 97% were actually Pneumonia.
* Out of all the x-rays that were Pneumonia x-rays, the model correctly predicted 99% of them
* Our model has a high f1 score, indicating incredible performance on classifying x-ray images as Pneumonia.


Already compared to our previous DNN models (including the initial model), our testing accuracy is higher than our training accuracy, signifying a slight decrease in overfitting. We also immediately see training loss occuring in our initial model loss curve, which we did not see in our initial DNN model. Lastly, we can see that our validation accuracy is 75%, which isn't bad for an initial model, but could be higher. Our initial results look promising, so let's try tuning further to see if we can improve our model. 

## CNN Model With Dropout Regularization ##

**Implementing dropout regularization:**

![cnn dropout reg](https://user-images.githubusercontent.com/45251340/216480307-a82831ed-5ba5-4eef-9cbc-1cd20c5fa8d1.JPG)

**Our epochs:**

![cnn dropout reg epochs](https://user-images.githubusercontent.com/45251340/216480524-9a3fefb3-dc26-4d18-8496-82d95cfde6e9.JPG)

**Our training and testing accuracy:** 

![cnn dropout reg accuracy](https://user-images.githubusercontent.com/45251340/216480558-ad516727-31a4-4d60-82a1-b93026ed6536.JPG)

**Our accuracy and loss curves:**

![cnn dropout reg curves](https://user-images.githubusercontent.com/45251340/216480570-876857f1-75c2-4ec8-81ff-2c3678de961b.JPG)

**Our training and testing accuracy are now almost the same, which is an improvement. Our training and testing loss are also extremely close in value. However our validation accuracy is still relatively low. Let's see what happens if we decrease the number of epochs to around 10.**

## CNN Model With Decreased Epochs##
**We used the same model as previous, but changed our number of epochs from 20 to 10.**

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
