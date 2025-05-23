# GAN Image Generator
GAN stands for Generative Aversarial Network.
This technique for image generation uses two types of networks to train; a discriminator and a generator.
- Made by [Emmett Jaakkola](https://www.linkedin.com/in/emmett-jaakkola-234667208/) and [Kyler Nikolai](https://www.linkedin.com/in/kylernikolai/).
## Discriminator
Determines if an input image is from the original dataset. Output is a probability from 0 to 1, where 0 is if it isn't in the dataset and 1 if it is.
## Generator
Generates images from an input noise vector. Tries to generate images that look like ones from the original dataset to fool the discriminator.
## Results
Most are not great, but we've seen a lot of improvement with the later models. Check out some of these images we were able to generate!
## Latest Version
Our latest version generates 128x128 images with three color channels. The model was trained on the
[Flickr Dataset](https://github.com/NVlabs/ffhq-dataset). This contained
higher quality images with less artifacts. This code has a slider that allows you to transition between two different random vectors shown [here](#sliders).
Using the `trainNN` function, you can run the model yourself and view the generated images.
Here is how you use the `trainNN` function:
- `epochs` - Set to 0 to run the model without training. Every epoch represents one pass through the dataset of 70,000 images.
- `batch_size` - The number of images we train the model on at a time. We found that 128 works best.
- `lr` - How much the model parameters are adjusted after each iteration. Lr = 0.0002 works well for Adam Optimizer.
- `save_time` - The number of epochs before each save.
- `save_dir` - The file where the model is saved and loaded from.
- `slide` - Set true to use sliders and false to generate images.

### Generated Images
<table>
  <tr>
    <td><img src="Images/face4.png" width="100%"/></td>
    <td><img src="Images/face5.png" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="Images/face6.png" width="100%"/></td>
    <td><img src="Images/face7.png" width="100%"/></td>
  </tr>
</table>

### Sliders <a id="sliders"></a>
<table>
  <tr>
    <td><img src="Images/slider0.gif" width="270px"/></td>
    <td><img src="Images/slider1.gif" width="270px"/></td>
  </tr>
</table>