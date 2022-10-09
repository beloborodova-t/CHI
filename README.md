# CHI
We present epithelium RoI identification based on automatic bounding boxes (bb) construction and SSE extraction. Further classification of the extracted epithelial fragments based on DenseNet made it possible to effectively identify the SSE RoI in cervical histology images (CHI). The design brings significant improvement to the identification of diagnostically significant regions. For this research, we created two CHI datasets, the CHI-I containing 171 color images of the cervical histology microscopy and CHI-II containing 1049 extracted fragments of microscopy, which are the most considerable publicly available SSE datasets.  
General pipeline of cervical digital histology image processing and RoI identifica-tion enriched with a new procedure of automatic bb construction embedded in this process. We start with data processing of cervical slide images to achieve out edge of SSE. Then we introduce the procedure of automatic bb construction. Third stage includes bb decomposition on patches and classification. Finally, we perform epithelium RoI identification and assessment. 
## Preprocessing stage includes: 
- (a) converting the RGB color model to the grayscale model, 
- (b) blurring the image according to Gaussian, 
- (c) image binarization using thresholding function, 
- (d) cervical contour detection. 
## Bounding box (bb) construction procedure includes:
- (a) shaping piecewise curves,
- (b) shaping rectangles, which are the basis of bb,
- (c) bb extraction

Parameters were used for images preprocessing and bbs extraction listed in [this section](Data/CHI-I/README.md).

## Classification

After bb extraction, a new dataset of histological image fragments is created. Depending upon the presence or absence of SSE, all extracted fragments are marked by expert with a positive class label in there any SSE, and a negative class label in the absence of SSE in the bb. The resulting dataset is used to train and test [DenseNet](https://arxiv.org/abs/1608.06993). 
The DenseNet is trained for 100 epochs with an Adam optimizer with a learning rate of 0.0001 under early stopping conditions. Training parameters: batch size = 128, patch size = 224. Weights are used to balance the simulation results to obtain a correct model. The model is run on [PyTorch platform](https://pytorch.org/) using nVidia GeForce RTX 2060 Super with 8GB of memory. Model training completed under 5 hours.

Trained model was used on 2 validation cervical histology slides which haven`t been used in model training and testing. initial images and images with costracted bb is shown in figure below.

![image](https://user-images.githubusercontent.com/53811556/193571788-d3b62c0f-ef1a-45cf-a2e6-1071ea35f209.png)

And the validation result presents in the next figure.

![image](https://user-images.githubusercontent.com/53811556/193572217-0a0c849b-0fe4-4a66-a126-7036e21cbf0a.png)
![image](https://user-images.githubusercontent.com/53811556/193572243-0a3a0cd7-03c7-4101-a81d-1096a95ee07e.png)

True Positive and True Negative patches classification marked in blue, False Positive and False Negative patches classification marked in red.
Comparison of RoIs are annotated by experts and extracted with proposed approach presented in figure below.

![image](https://user-images.githubusercontent.com/53811556/193572724-2fe23715-0565-45fb-85e3-5a76772db2fa.png)

## How to Cite

If you find this work helpful, please cite it as "[Biloborodova T., Lomakin S., Skarga-Bandurova I., Krytska Y. Region of Interest Identification in the Cervical Digital Histology Images. In: Marreiros, G., Martins, B., Paiva, A., Ribeiro, B., Sardinha, A. (eds) Progress in Artificial Intelligence. EPIA 2022. Lecture Notes in Computer Science, vol 13566, 2022, p. 133–145. Springer, Cham. https://doi.org/10.1007/978-3-031-16474-3_12.](https://link.springer.com/chapter/10.1007/978-3-031-16474-3_12)"
