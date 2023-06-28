# Flood Segmentation Using UNet

## Inspirations

Houston frequently encounters flooding, which leads to significant damages, especially during the hurricane season. The primary goal of this project is to utilize machine learning and computer vision to accurately detect affected areas in images, offering valuable insights for disaster response and mitigation endeavors.

#### Houston's flood in 2018
![Houston](images/houston.png)

## Datasets

Obtained from Kaggle

#### Acknowlegement
```
https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation
```

#### Preprocessing 

The dataset includes 290 images and their correspoding mask, which can be considered small. To expand the dataset, we will utilize augmentation techniques. The size of training set increase from 185 to 1183 images after augmentation.


![Augment](images/augment.png)

## Model Performance Comparations

#### Trained on original dataset

![Original](plots/Accuracy_on_Original_Dataset.png)

#### Trained on augmented dataset

![Augmentation](plots/Accuracy_on_Augmented_Dataset_Batch8.png)


## Test on Houston flood images

![Houston](images/houston_predicted.png)


