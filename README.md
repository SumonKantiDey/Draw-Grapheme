# Bengali.AI Handwritten Grapheme Classification [link](https://www.kaggle.com/c/bengaliai-cv19)
<img src="https://github.com/SumonKantiDey/Draw-Grapheme/blob/master/templates/static/image/grapheme.png" >

This project involves classifying handwritten characters of the Bengali alphabet which is broken down into three components for each grapheme, or character: `1.the root`, `2.the vowel diacritic`, `3. the consonant diacritic`. The goal is the create a classification model that can classify each of these three components of a handwritten grapheme. Click [here](https://www.kaggle.com/c/bengaliai-cv19) for more information. To know details about bengali grapheme check [Introductory Booklet](https://bengali.ai/wp-content/uploads/CV19-COCO-Grapheme.pdf).

## Methods Used
- Exploratory Data Analysis
- Image Preprocessing
- MultiOutPut Convolutional Neural Networks
- Hyperparameter Tuning


## Project Demo Video

An experimental app for the web that can capture  `1.the root`, `2.the vowel diacritic`, `3. the consonant diacritic` from Bengali handwritten grapheme. Check this [ipython](https://github.com/SumonKantiDey/Draw-Grapheme/blob/master/multioutput-cnn-101.ipynb) notebook to save the weight of the **multioutput CNN model**.
<p align="center">
<img src="https://j.gifs.com/p8vPkN.gif" width="400" height="400" />
</p>


## Technologies

* Deep Learning

     - Convolution Neural Network (CNN)
     - Python, keras, tensorflow, opencv, numpy

* Web

     - Flask (python web framework)
     - Jquery, Ajax, Bootstrap


## CNN Model Summary
Model: "model_1"
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 64, 64, 1)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 64, 64, 32)   320         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, 64, 32)   9248        conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 64, 32)   9248        conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 64, 64, 32)   128         conv2d_3[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 32, 32, 32)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 32)   25632       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 64)   18496       conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 64)   36928       conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 64)   36928       conv2d_6[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 64)   256         conv2d_7[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 16, 16, 64)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 16, 16, 64)   102464      max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 16, 16, 64)   256         conv2d_8[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 16, 16, 64)   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 128)  73856       dropout_1[0][0]                  
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 128)  147584      conv2d_9[0][0]                   
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 128)  147584      conv2d_10[0][0]                  
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 16, 16, 128)  512         conv2d_11[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 8, 8, 128)    0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 8, 8, 128)    409728      max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 8, 8, 128)    512         conv2d_12[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 8, 8, 128)    0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 8, 8, 256)    295168      dropout_2[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 8, 8, 256)    590080      conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 8, 8, 256)    590080      conv2d_14[0][0]                  
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 8, 8, 256)    1024        conv2d_15[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 4, 4, 256)    0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 4, 4, 256)    1638656     max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 4, 4, 256)    1024        conv2d_16[0][0]                  
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 4, 4, 256)    0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 4096)         0           dropout_3[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         4195328     flatten_1[0][0]                  
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 1024)         0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 512)          524800      dropout_4[0][0]                  
__________________________________________________________________________________________________
dense_root (Dense)              (None, 168)          86184       dense_2[0][0]                    
__________________________________________________________________________________________________
dense_vowel (Dense)             (None, 11)           5643        dense_2[0][0]                    
__________________________________________________________________________________________________
dense_consonant (Dense)         (None, 7)            3591        dense_2[0][0]                    
==================================================================================================
Total params: 8,951,258
Trainable params: 8,949,402
Non-trainable params: 1,856
__________________________________________________________________________________________________
```

## Installation

To build and run the app, first of all clone this project.

`pip install -r requirements.txt` run this command to install the dependencies of this project

`python main.py` to run this project
