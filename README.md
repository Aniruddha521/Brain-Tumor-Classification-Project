# Brain Tumor Classification using CNN

## Project Overview:

This project aims to develop a Convolutional Neural Network (CNN) model for the classification of brain tumors based on medical imaging data. The deep learning model is designed to analyze magnetic resonance imaging (MRI) scans and accurately identify whether a given image contains a tumor or not.

## Introduction:

Medical image analysis plays a crucial role in the early detection and diagnosis of various conditions, including brain tumors. This project leverages deep learning techniques, specifically CNNs, to automate the classification of brain tumor images, assisting medical professionals in their diagnostic processes.

## Installation:

All the packages which are are given in required_packages.txt.

run `pip install -r required_packages.txt` in terminal

By using this command we can install all required packages(I have installed all requied packages in a virtual enviroment. It is recommended to do the same).

## Dataset

The dataset used for training and testing the model is taken from [https://www.kaggle.com]. The dataset is divided into training and testing sets to evaluate the model's performance accurately.Both training and testing sets are further divided into four subsets. Among which one set is for no tumor and the rest of the sets are for different types of brain tumor`(glioma tumor, meningioma tumor and pituitary tumor)`.

## Model Architecture

The CNN architecture used in this project is designed to extract features from brain tumor images efficiently. The architecture is inspired fron Googlenet and Resnet architecture. It uses both the Inception module as well as Residual blocks, where each residual block is generally divided into three section first and the last section contains inception module and the middle section contains two convucational layers of `(3X3 + 1(s))` and `(5X5 + 1(s))` respectively.

`Note:- The models are trained with preprocess data. So make sure that before predicting any image you must preprocess it.`

## Training

To train the model on your dataset, run ` python Train_model.py`  or you can go through Brain Tumor Classification.ipynb notebook to train and test models.

## Results

`model_3.h5` has scored highest accuracy until now with 90% above in both training and validation sets.
