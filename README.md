# Image-Classification

## Business Goal ##
We'll be working with Palomar Medical Center, located in Escondido, California. Initially using a deep neural network, then moving onto a Convolutional Neural Network, we build a model that takes patient X-rays and identifies if the patient has Pneumonia or not. Creating a successful model will help the hospital doctors make a quicker diagnosis by reducing (if not eliminating totally) the time spent on consulting X-rays before meeting with patients. Once proven successful, It may also help to generalize over to other diseases and ailments that are able to be diagnosed via x-ray, such as broken bones and certain types of cancer.

## Data ##
The data comes from a kaggle dataset of chest X-ray images. There are 5,856 images in total, and the data is split into train, test and validation folders. Each folder is further split into folders that contain Pneumonia and non Pneumonia X-rays. The images are split up in the folders as follows: 


![training images](https://user-images.githubusercontent.com/45251340/216470311-bab2ac1f-fd4d-46d2-be38-2fc4ad43e7a9.JPG)


![testing images](https://user-images.githubusercontent.com/45251340/216470408-b3674dbe-f612-4778-8136-04066fbe1a1e.JPG)


![validation images](https://user-images.githubusercontent.com/45251340/216470461-c531972d-4403-4f35-a633-4d6d49dd1b42.JPG)


**From our image distributions, we can see that there is a data imbalance, with there being more Pneumonia x-rays than normal x-rays, which we'll address in our CNN model**

## Starting off ##
First we start off by taking all the input images from our directories and reshape them into a 200 x 200 size. After that, we create our training, testing and validation data sets. Finally we reshape our datasets for use in our model. Our initial sets had the following shapes:

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

**Our model had:**
* A training accuracy of 92%
* A testing accuracy of 72% 
* A validation accuracy of 75%

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

**Classification report and confusion matrix:**

![cnn initial model classification report and cm](https://user-images.githubusercontent.com/45251340/219891454-8755380b-4f4f-43a4-8a4a-291f61ad66ba.JPG)

**Our model had:**
* A training accuracy of 96%
* A testing accuracy of 97% 
* A validation accuracy of 75%

**Let's take a look at our classification report results:**

For our test model class 1 (meaning a chest x-ray is classified as Pneumonia) we have a precision score of .97, a recall score of .99, and an f1 score of .98, meaning:

* Out of all the x-rays that the model classified as Pneumonia, 97% were actually Pneumonia.
* Out of all the x-rays that were Pneumonia x-rays, the model correctly predicted 99% of them
* Our model has a high f1 score, indicating incredible performance on classifying x-ray images as Pneumonia.

**Let's take a look at our confusion matrix results:**
* 160 x-rays were correctly classified as being normal x-rays
* 6 x-rays were wrongly classified as not being Pneumonia x-rays
* 12 x-rays were wrongly classified as being Pneumonia x-rays
* 446 x-rays were correctly classified as being Pneumonia x-rays


Already compared to our previous DNN models (including the initial model), our testing accuracy is higher than our training accuracy, signifying a slight decrease in overfitting. We also immediately see training loss occuring in our initial model loss curve, which we did not see in our initial DNN model. Lastly, we can see that our validation accuracy is 75%, which isn't bad for an initial model, but could be higher. Our initial results look promising, but we tried various other tuning methods beore finally arriving at our final model. 

## Final CNN Model ##
After various tuning methods were used, we arrived at our final improved CNN model. Our final model implemented both dropout regularization, and a decreased number of epochs. 

![final cnn model summary](https://user-images.githubusercontent.com/45251340/219893490-22fff15c-9338-40b1-bd8f-c8651ddc8abe.JPG)

**Our accuracy and loss curves:**

![final cnn model accuracy and loss curves](https://user-images.githubusercontent.com/45251340/219893750-d7c3f977-3ac7-4987-b708-7de2ef2eef90.JPG)

**Classification report and confusion matrix:**

![final cnn model classification report and cm](https://user-images.githubusercontent.com/45251340/219893980-0e4bbc5b-5ce7-460f-93d8-f0743ccae03c.JPG)

**Our model had:**

* A training accuracy of 86%
* A testing accuracy of 88%
* A validation accuracy of 81%

**Let's take a look at our classification report results:** 

For our test model class 1 (meaning a chest x-ray is classified as Pneumonia) we have a precision score of .99, a recall score of .85, and an f1 score of .91, meaning:

* Out of all the x-rays that the model classified as Pneumonia, 99% were actually Pneumonia.
* Out of all the x-rays that were Pneumonia x-rays, the model correctly predicted 85% of them
* Our model has a high f1 score, indicating great performance on classifying x-ray images as Pneumonia.

**Let's take a look at our confusion matrix results:**
* 169 x-rays were correctly classified as being normal x-rays
* 70 x-rays were wrongly classified as not being Pneumonia x-rays
* 3 x-rays were wrongly classified as being Pneumonia x-rays
* 382 x-rays were correctly classified as being Pneumonia x-rays


We can now see that the difference between our training and testing accuracy is slightly higher than it's been in our past DNN and CNN iterations, which indicates a further decrease in overfitting. Our validation accuracy is also the highest it has been during our tuning. This, along with a more realistic recall score shows significant improvement from our initial/previous DNN and CNN models.


## Conclusion ##

The goal was to work with Palomar Medical Center to build an image classification model that analyzes patient X-rays and identifies if the patient has Pneumonia or not. This will help the hospital doctors make a quicker diagnosis, and reduce the time spent on consulting X-rays before meeting with patients. I initially used a Deep Neural Network (DNN), then moved onto a Convolutional Neural Network (CNN).

I initially started with a DNN model, but found that the initial model was greatly overfitting, and needed work. Next, I tuned the model multiple times, by adjusting things like the number of layers, and the number of epochs of the model. I also tried implementing dropout regularization to further reduce overfitting. Despite my efforts, the model kept overfitting, and the accuracy was consistently low. As mentioned earlier, this could be due to the imbalance of data (more Pneumonia images than normal images), and also the type of neural network used (DNN). Therefore, I decided to try and use a different approach and switch to a Convolutional Neural Network (CNN).

My results drastically improved when switching over from a DNN to a CNN. The initial CNN model already performed better than the initial DNN (as well as the other DNN models). Through further tuning, I was able to further reduce overfitting (as seen in the increased difference between our training and testing accuracy) and obtain a more realistic recall score. These results can be seen in our final CNN model summary above. However there are still some things that I'd like to improve upon. I'd like to see a higher testing accuracy, as well as a higher validation accuracy. For testing accuracy I'd like to see a value around 95%, and for validation accuracy I'd like to see a value between 90 and 95%. This can be obtainable by increasing the amount of images in each set, as well as trying out more tuning methods/combination of tuning methods.

**Recommendations**

There are a few recommendations I have for using an improved model in a hospital setting. For starters, it would be beneficial to use this model in the same setting where the x-ray is taken. This way, the x-ray can immediately be analyzed by the model, vs having to print out and develop the x-ray, then have a doctor analyze the x-ray. Initially the doctor may need to double check the analysis, but the goal is to create a completely reliable model that can diagnose an x-ray without needing a doctor to check. Another recommendation would be to try out different sized x-ray formats. Different sizes may be processed at different speeds, or processed more efficiently, so it would be beneficial to try to find the right size.

If these technical and hospital setting recommendations are implemented, than a model can be created that will greatly reduce the time spent by doctors analyzing x-rays and even potentially remove the need entirely. Removing this need for evaluation and analysis will allow doctors to spend more time with their patient(s), and can even be revolutionary in diagnosing other ailments such as broken bones, bone cancer, and even breast cancer.

