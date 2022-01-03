# mask-detector

A convolutional neural network that can detect how people are wearing a mask, given a photo centered on their face.

## Usage
This model has been deployed using a Rest API with Flask. Follow these steps to get a classification of a custom photo:

* Use a client that can send HTTP requests (e.g. Insomnia or cURL) to query https://ml-api-1024.herokuapp.com/ with a POST request. 
* Use a multipart form and attach one (or multiple) photos. For best results, make sure the face takes up the whole image.
* The Rest API will return a json object (Python dictionary) containing the classification for each photo for each file (indexed by filename).

![image](https://user-images.githubusercontent.com/17630457/147899457-0c7a2ccb-a653-4e05-8615-c7c7bb9f0d97.png)

### Classifications
* `incorrect_mask` means the mask is not worn properly.
* `with_mask` represents a correctly worn mask.
* `without_mask` represents no mask on the face.

## Neural Network Architecture
Transfer learning from the VGG19 convolutional network was used, with extra preprocessing being added
at the beginning and end of the network to improve accuracy and reduce training time. Strategies included randomly rotating + zooming images to reduce overfitting,
and pooling the global average of a layer instead of simply flattening a volume. 

Overall, the model was trained to a validation accuracy of 95%. For details about the process, check `mask.ipynb`. 

## Tech Stack
* Tensorflow, Keras, Python, Flask

## Dataset
Used [this dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) on Kaggle. The bounding boxes for faces were extracted from the xml files corresponding to each image and then cropped/resized accordingly.
