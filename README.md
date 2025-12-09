
# Tutorial

This lab (Pratical Work/tutorial/TP) is for Master student of the course "Machine Learning and Images".

From a video of a source person and another of a person, the objective is to generate a new video of the targeted person performing the same movements as the source. We test several NN than learn how to generate images of the targeted person according to a skeleton pose.

[See the course main page with the description of this tutorial/TP](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

## How to run the code using the trained network : 
The first step is to clone this repository in order to create a copy you can modify at your will.

### 1 - Choice of the model to execute : 
**Our GAN models** : 
You can select the model you want to execute from the *"GenGAN.py"* file => *"__init__"* function (lign 78).
From there, simply edit the name of the file containing the model you want to execute. These files are available in *src/data/Dance*

**Our Vanilla models** : 
Similarily, you can select the models you want to execute from *"GenVanillaNN.py"* file => lign 248 or 255.
The available models are also stored in *src/data/Dance*.

Our pretrained models are : 
- *DanceGenGAN.pth* : our most up-to-date GAN version, with attention mechanism and skip-connections
- *DanceGenGAN (without Attention).pth* : 1st version of this GAN model, without attention mechanism
- *DanceGenVanillaFromSke26.pth* : model compatible with the *"GenVanillaNN.py"* file

## 2 - Code execution : 
The file to execute is **DanceDemo.py**. 
In its *__main__* function (lign 67), select the following parameters :

**GEN_TYPE parameter** : 
- 1 : to execute NEAREST - *GenNearest.py* algorithm
- 2 : to execute VANILLA_NN method - *GenVanillaNN.py* (direct neural network)
- 3 : to execute VANILLA_NN_Image method - *GenVanillaNN.py* (Neural network with the skeleton as input)
- 4 : to execute the GAN method - *GenGAN.py*
