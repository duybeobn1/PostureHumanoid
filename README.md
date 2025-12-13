# Posture-guided image synthesis of a person

**Repository:** [https://github.com/duybeobn1/PostureHumanoid](https://github.com/duybeobn1/PostureHumanoid)

From a video of a source person and another of a target person, the objective is to generate a new video of the targeted person performing the same movements as the source. We test several Neural Networks (NN) that learn how to generate images of the targeted person according to a skeleton pose.

[See the course main page with the description of this tutorial/TP](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

## 00 - Our demo



## A - How to run the code using the trained network

The first step is to clone this repository in order to create a copy you can modify at your will.

### 1 - Choice of the model to execute
**Our GAN models**:
You can select the model you want to execute from the *"GenGAN.py"* file => `__init__` function.
From there, simply edit the name of the file containing the model you want to execute. These files are available in *src/data/Dance*.

**Our Vanilla models**:
Similarly, you can select the models you want to execute from the *"GenVanillaNN.py"* file.
The available models are also stored in *src/data/Dance*.

Our pretrained models are:
- **DanceGenGAN.pth**: *Final Version* (350 epochs). Uses **Smooth U-Net** (Upsampling + Conv) + Attention on High-Res (256x256) data. Best quality with reduced noise.
- **DanceGenGAN150final.pth**: *Intermediate Version* (150 epochs). Uses Smooth U-Net. Good quality but less refined than the final version.
- **DanceGenGAN10.pth**: *Early Checkpoint* (10 epochs). Uses Smooth U-Net. Shows the early stages of training stability.
- **DanceGenGAN256-256-1.pth**: *Comparison Version* (150 epochs). Trained on High-Res data but **without** Smooth U-Net (uses standard Deconvolution). Useful to demonstrate checkerboard artifacts compared to the smooth version.
- **DanceGenGAN50.pth**: *Comparison Version* (50 epochs). Trained on High-Res data **without** Smooth U-Net.
- **DanceGenGAN_64_64_2000epochs.pth**: *Baseline Version* (2000 epochs). Old architecture trained on Low-Res (64x64) data. No more executable on the last version of our code, you can use it by pulling the "UPDATE : 2000 epoches training" commit (29/11/2025).
- **DanceGenVanillaFromSke26.pth**: Model compatible with the *"GenVanillaNN.py"* file (Vector input).

### 2 - Code execution
The file to execute is **DanceDemo.py**.
In its `__main__` function, select the following parameters:

**GEN_TYPE parameter**:
- 1: Execute NEAREST - *GenNearest.py* algorithm
- 2: Execute VANILLA_NN method - *GenVanillaNN.py* (direct neural network with vector input)
- 3: Execute VANILLA_NN_Image method - *GenVanillaNN.py* (Neural network with the skeleton image as input)
- 4: Execute the GAN method - *GenGAN.py* (High-Res U-Net + Attention)

**Video selection**:
In order to choose the video from which we're going to copy the movements, edit the following line in `DanceDemo.py`:
```python
ddemo = DanceDemo("../data/karate_full.mp4", GEN_TYPE)
````

## B - How to train and save a network

**For GenGAN.py**:

  - **Configuration**: Inside the `if __name__ == '__main__':` block:
      - Set `loadFromFile = False` to train a new model from scratch.
      - Set `loadFromFile = True` to continue training an existing model.
      - Choose the number of epochs: `gen.train(n_epochs=500)` (We recommend 400-500 for high-res results).
  - **Optimization**: The code automatically detects **Apple Silicon (M1/M2/M3/M4)** chips and uses **MPS (Metal Performance Shaders)** acceleration for significantly faster training compared to CPU.

**For GenVanillaNN.py**:
Similarly:

  - Switch the value of `train` to `True` in the main block.
  - Choose if you want to train a model from scratch or load a pre-trained one: `gen = GenVanillaNN(targetVideoSke, loadFromFile=False)`.
  - Choose your number of epochs: `n_epoch = 200`.

## C - Implementation & Technical Evolution

Our project evolved through three distinct phases, moving from a low-resolution baseline to a high-fidelity generative model. Below is the detailed breakdown of our technical contributions.

### 1\. Phase I: Enhanced Low-Resolution Baseline ($64 \times 64$)

Our initial experiments operated on $64 \times 64$ images using the provided `VideoSkeleton` class. We quickly identified that the basic Encoder-Decoder architecture (Bottleneck) failed to preserve structural details. To address this, we implemented a **U-Net** architecture (saved as *DanceGenGAN.pth*):

  * **Skip Connections:** We added 3 skip connections, concatenating features from the Encoder (`enc1`, `enc2`, `enc3`) directly to the corresponding Decoder layers. This allowed the network to recover spatial details lost during downsampling.
  * **Self-Attention:** We integrated a Self-Attention layer (SAGAN) at the $8 \times 8$ resolution block. This allowed the network to model global dependencies (e.g., limb coordination) that standard convolutions miss.
  * **Result:** While the structure was correct, the output remained too blurry for practical use due to the low resolution of the input data.

### 2\. Phase II: High-Resolution Data Pipeline

To enable high-quality generation, we completely overhauled the data extraction pipeline in `VideoSkeleton.py`:

  * **Resolution Upgrade:** We modified the extraction logic to process source videos at a width of **512px** (up from 256px). This ensures that the cropped subject maintains high-frequency details (eyes, fingers) before being resized to the target $256 \times 256$ input for the network.
  * **Robust Cropping (`crop_with_padding`)**: We identified a critical issue where standard cropping would crash or distort the aspect ratio when the subject moved to the edge of the camera frame. We implemented a custom `crop_with_padding` function that calculates the missing area and intelligently pads it with black pixels. This guarantees that every training sample has consistent dimensions without geometric distortion.

### 3\. Phase III: The "Smooth" U-Net ($256 \times 256$)

We successfully trained our Phase I architecture on the new high-resolution data. While the sharpness improved significantly, we observed severe **"Checkerboard Artifacts"** (grid-like noise) in the background. We diagnosed this as a side-effect of `ConvTranspose2d` (Deconvolution) layers operating at high resolution.

To eliminate these artifacts, we engineered our final model (**Smooth U-Net**, saved as *DanceGenGAN\_256\_256.pth*):

  * **Upsampling Strategy:** We replaced all `ConvTranspose2d` layers with a composite block of **`nn.Upsample(scale_factor=2, mode='bilinear')` + `nn.Conv2d`**. This decoupling of upsampling and convolution mathematically eliminates the uneven overlap that causes grid noise.
  * **Adaptive Discriminator**: To support the resolution jump, we added an `AdaptiveAvgPool2d(1)` layer at the end of the Discriminator. This makes the critic resolution-agnostic, allowing us to train on $256 \times 256$ images without altering the dense layers.
  * **Optimized Training**: We utilized the **Wasserstein GAN (WGAN-GP)** objective with Gradient Penalty and `InstanceNorm2d` to ensure training stability at this higher complexity.

**Final Result**: This architecture retained the sharpness from the skip connections and attention mechanism but produced significantly cleaner, noise-free backgrounds compared to the standard deconvolution approach.

