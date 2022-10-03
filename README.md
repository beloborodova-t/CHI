# CHI
We present epithelium RoI identification based on automatic bounding boxes (bb) construction and SSE extraction. Further classification of the extracted epithelial fragments based on DenseNet made it possible to effectively identify the SSE RoI in cervical histology images (CHI). The design brings significant improvement to the identification of diagnostically significant regions. For this research, we created two CHI datasets, the CHI-I containing 171 color images of the cervical histology microscopy and CHI-II containing 1049 extracted fragments of microscopy, which are the most considerable publicly available SSE datasets.  
General pipeline of cervical digital histology image processing and RoI identifica-tion enriched with a new procedure of automatic bb construction embedded in this process. We start with data processing of cervical slide images to achieve out edge of SSE. Then we introduce the procedure of automatic bb construction. Third stage includes bb decomposition on patches and classification. Finally, we perform epithelium RoI identification and assessment. 
## Preprocessing stage includes: 
- converting the RGB color model to the grayscale model, 
- blurring the image according to Gaussian, 
- image binarization using thresholding function, 
- cervical contour detection. 
## Bounding box (bb) construction procedure includes:
- shaping piecewise curves,
- shaping rectangles, which are the basis of bb,
- bb extraction
## Classification
After bb extraction, a new dataset of histological image fragments is created. Depending upon the presence or absence of SSE, all extracted fragments are marked by expert with a positive class label in there any SSE, and a negative class label in the absence of SSE in the bb. The resulting dataset is used to train and test DensNet.
Trained model was used on 2 validation cervical histology slides which haven`t been used in model training and testing. initial images and images with costracted bb is shown in figure below.
![image](https://user-images.githubusercontent.com/53811556/193571788-d3b62c0f-ef1a-45cf-a2e6-1071ea35f209.png)
