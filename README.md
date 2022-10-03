# CHI
We present epithelium RoI identification based on automatic bounding boxes (bb) construction and SSE extraction. Further classification of the extracted epithelial fragments based on DenseNet made it possible to effectively identify the SSE RoI in cervical histology images (CHI). The design brings significant improvement to the identification of diagnostically significant regions. For this research, we created two CHI datasets, the CHI-I containing 171 color images of the cervical histology microscopy and CHI-II containing 1049 extracted fragments of microscopy, which are the most considerable publicly available SSE datasets.  
General pipeline of cervical digital histology image processing and RoI identifica-tion enriched with a new procedure of automatic bb construction embedded in this process. We start with data processing of cervical slide images to achieve out edge of SSE. Then we introduce the procedure of automatic bb construction. Third stage includes bb decomposition on patches and classification. Finally, we perform epithelium RoI identification and assessment. 
/Preprocessing stage includes: 
(a) converting the RGB color model to the grayscale model, 
(b) blurring the image according to Gaussian, 
(c) image binarization using thresholding function, 
(d) cervical contour detection. 
/Bounding box (bb) construction procedure includes:
(a) shaping piecewise curves,
(b) shaping rectangles, which are the basis of bb,
(c) bb extraction
/Classification
