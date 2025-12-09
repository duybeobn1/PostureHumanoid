
# Tutorial

This lab (Pratical Work/tutorial/TP) is for Master student of the course "Machine Learning and Images".

From a video of a source person and another of a person, the objective is to generate a new video of the targeted person performing the same movements as the source. We test several NN than learn how to generate images of the targeted person according to a skeleton pose.

[See the course main page with the description of this tutorial/TP](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

## A - How to run the code using the trained network : 
The first step is to clone this repository in order to create a copy you can modify at your will.

### 1 - Choice of the model to execute : 
**Our GAN models** : 
You can select the model you want to execute from the *"GenGAN.py"* file => `__init__` function (Line 78).
From there, simply edit the name of the file containing the model you want to execute. These files are available in *src/data/Dance*

**Our Vanilla models** : 
Similarily, you can select the models you want to execute from *"GenVanillaNN.py"* file => Line 248 or 255.
The available models are also stored in *src/data/Dance*.

Our pretrained models are : 
- *DanceGenGAN_256_256.pth* : our most up-to-date GAN version, with attention mechanism and skip-connections, the update skeleton version and self.image_size = 256
- *DanceGenGAN.pth* : our previous GAN version, with attention mechanism and skip-connections only.
- *DanceGenGAN (without Attention).pth* : 1st version of this GAN model, without attention mechanism
- *DanceGenVanillaFromSke26.pth* : model compatible with the *"GenVanillaNN.py"* file

### 2 - Code execution : 
The file to execute is **DanceDemo.py**. 
In its `__main__` function (Line 67), select the following parameters :

**GEN_TYPE parameter** : 
- 1 : to execute NEAREST - *GenNearest.py* algorithm
- 2 : to execute VANILLA_NN method - *GenVanillaNN.py* (direct neural network)
- 3 : to execute VANILLA_NN_Image method - *GenVanillaNN.py* (Neural network with the skeleton as input)
- 4 : to execute the GAN method - *GenGAN.py*

**Video selection** : 
In order to choose the video from which we're going to copy the movements, edit the following line : 
*ddemo = DanceDemo("../data/karate_full.mp4", GEN_TYPE)*

## B - How to train and save a network : 

**For GenGAN.py** : 
- Line 260 : switch the variable *TRAIN_MODE*  to *True*
- Line 265 : you can switch the value of *loadFromFile* in order to further the training of an already pre-trained model : `gen = GenGAN(targetVideoSke, loadFromFile=False)`. Otherwise, you will create and train a new model from scratch
- Line 269 : choose the number of epochs of your training : `gen.train(n_epochs=2000)`

**For GenVanillaNN.py** : 
Similarly : 
- Line 329 : switch the value of *train* to *True*
- Line 346 : choose if you want to train a model from scratch of load a pre-trained one : `gen = GenVanillaNN(targetVideoSke, loadFromFile=False)`
- Line 328 : choose your number of epochs : `n_epoch = 200`

## C - What we implemented 
Our efforts were focussed on the *GenGAN.py* file. We built the following architecture : 

**Discriminator** : 
Simple discriminator, which consists in a Convolutional Neural Network (CNN). With 2D convolutions progressively computing up to 512 features, it generates a Critic Score prediction.
- High score (positive) = The image is considered realistic (close to the real data distribution)
- Low score (negative) = The image is considered unrealistic (close to the fake/generated data distribution).
It takes in entry an image (of size $3 \times 64 \times 64$), and as output a single scalar value (the Critic Score).

**Generator** : 
The model is parameterized in the file *GenVanillaNN*, within the function class `GenNNSkeImToImage(nn.Module)`.
Its features include : 
- 4-layer Encoder, which performs downsampling from 3 to 512 features.
- 3-layer Decoder
- skip connections : 2 skip connections
  - from layer 3 of encoder => joined with output of layer 1 of encoder by concatenation
  - from layer 1 of encoder => joined with output of layer 3 of encoder by concatenation
- self-attention layer of 1st decoding layer.

**Image Size & Skeleton architecture**
Raising the *self.image_size* parmater has shown to improve the model's performances. However, is significantly inscreases the length of each training epoch.
We chose `self.image_size = 256`for our final version of the model (*DanceGenGAN_256_256.pth*)

We also updated the *VideoSkeleton.py* file :
- changed the default newVideoWidth from 256 to 512 (both in the init method and the main block).
- implemented and used the crop_with_padding helper function inside cropAndSke. When a person moves to the very edge of the camera frame, a standard crop would either fail (crash) or distort the image because the required area "doesn't exist." This function adds black pixels (padding) to fill the void, ensuring every extracted image has the exact same dimensions without crashing.
- updated the creation of the empty canvas to use self.ske_height_crop and self.ske_width_crop. This ensures that when run VideoSkeleton.py to test data, the window displays the full, high-resolution cropped images, allowing to visually verify that the data is indeed sharp before training. ( U-net GAN)

