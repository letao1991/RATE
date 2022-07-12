Recurrent Affine Transform Enocoder (RATE) for Image Representation

Data Set: Traffic Sign

environment: pytorch 1.8, cuda 11.1

1. Download dataset:
1.1 download training data: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip

1.2 downlaod testing data: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip

1.3 extract the downloaded data to the current folder

2. test with pretrained model: 
download link: https://drive.google.com/file/d/1BxIXY9IfLvXq3ppfo57O5u90F_Gx1jN1/view?usp=sharing

python test_alignment.py

The distorted and aligned images will be stored in "images" folder

3. train from scratch (takes around 10 mins on RTX3090):
python RATE_encoder.py

The trained model will be saved in the current folder (default: "discriminator_rst_6000").

You can then test the trained model with: 
python test_alignment.py

