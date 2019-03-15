## Neural Style Transfer in Rust

This example shows how to implement
[Neural Style Transfer](https://arxiv.org/abs/1508.06576) using PyTorch and Rust,
it is very close to the
[Neural Transfer using PyTorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
which contains far more details.

The Neural Style Transfer algorithm takes as input a content image and a style image and
produces some new images looking similar to the content image but using the style
from the second image.

### Running the Code
This example uses a pre-trained VGG-16 convolutional network, the weights can be found in
[vgg16.ot](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg16.ot).

Then the example can be run with the following command, this will create some new images in
the current directory.

```bash
cargo run --example neural-style-transfer style.jpg content.jpg vgg16.ot
```

Here is some example content and output images using [the Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night)
as a style image.

![20 Fenchurch Street](https://raw.githubusercontent.com/LaurentMazare/ocaml-torch/master/examples/neural_transfer/20fen.jpg)
![Starry 20 Fen](https://raw.githubusercontent.com/LaurentMazare/ocaml-torch/master/examples/neural_transfer/20fen-starry.jpg)

([original image source](https://commons.wikimedia.org/wiki/File:Walkie-Talkie_-_Sept_2015.jpg))

### Neural Style Transfer

The Neural Style Transfer algorithm works by gradient descent starting from
an initial image (we use the content image as a starting point but could also
use random noise).

This gradient descent aims at minimizing a loss composed of two parts: a style loss and a
content loss. The style loss evaluates how much our current image is similar to the style
image in terms of style, whereas the content loss focuses on whether our current image
has a content close to the reference content image.

In order to define `being similar` we use a pre-trained convolutional neural
network. This network is run on the current image as well as on the style and
content images.  Intuitively the features extracted by the lower layers are
more about style and the features extracted by the upper layers about contents.

The content loss is then computed using MSE on the extracted features for some
upper layers.  MSE helps ensuring that the extracted features are similar between
the two images at the same on-screen locations.

For the style loss we do not really care about spatial location of the features
in the style image. So rather than MSE we use some Gram matrix to define a loss
that is location independent.

### Implementation 1: Loading the Model and Images

Let's have a look at how we can code this in rust.
The first thing we do is create a device which can be either a cpu or a gpu depending on what is
available.

```rust
    let device = Device::cuda_if_available();
```

We then create a variable store on this device, this variable store will be used to hold
the pre-trained variables for the VGG-16 model.
Using this variable store we create the model and load the weights from the weight file
on disk. Finally we *freeze* the variable store ensuring that the model weights are
not modified when running some gradient descent later in the process.

```rust
    let mut net_vs = tch::nn::VarStore::new(device);
    let net = vgg::vgg16(&net_vs.root(), imagenet::CLASS_COUNT);
    net_vs.load(weights_file)?;
    net_vs.freeze();
```

Next we load the style and content image, and move them to the device.
Note that the `unsqueeze` method is used to add a batch-dimension of size 1 so that
the model can be run on these images.

```rust
    let style_img = imagenet::load_image(style_img)?
        .unsqueeze(0)
        .to_device(device);
    let content_img = imagenet::load_image(content_img)?
        .unsqueeze(0)
        .to_device(device);
```

### Implementation 2: Running the Model on the Content and Style Images

We now run the model on the style and content images. These calls returns some
vector containing the extracted features for each of the model layers.
```rust
    let style_layers = net.forward_all_t(&style_img, false, Some(max_layer));
    let content_layers = net.forward_all_t(&content_img, false, Some(max_layer));
```

As we will use gradient descent to optimize an image, we create a second
variable store to hold this image as a variable.  The initial value for this
variable is obtained by copying the content image.

```rust
    let vs = nn::VarStore::new(device);
    let input_var = vs.root().var_copy("img", &content_img);
```

### Implementation 3: Gradient Descent Loop

We use Adam for optimization: creating an optimizer requires to pass it the
variable store that we want to optimize on.

```rust
    let opt = nn::Optimizer::adam(&vs, LEARNING_RATE, Default::default())?;
```

Then we run our gradient descent loop. In this loop we compute the style
and content losses, sum them, and run an optimization step.

We also regularly print the current loss value and write a file containing the
optimized image.

```rust
    for step_idx in 1..(1 + TOTAL_STEPS) {
        // ... compute style_loss and content_loss ...
        let loss = style_loss * STYLE_WEIGHT + content_loss;
        opt.backward_step(&loss);
        if step_idx % 1000 == 0 {
            println!("{} {}", step_idx, f64::from(loss));
            imagenet::save_image(&input_var, &format!("out{}.jpg", step_idx))?;
        }
    }
```

In order to compute the losses, we first evaluate our model on the current image.
The style loss is then extracted from the style layers and the content loss from
the content layers.

```rust
        let input_layers = net.forward_all_t(&input_var, false, Some(max_layer));
        let style_loss: Tensor = STYLE_INDEXES
            .iter()
            .map(|&i| style_loss(&input_layers[i], &style_layers[i]))
            .sum();
        let content_loss: Tensor = CONTENT_INDEXES
            .iter()
            .map(|&i| input_layers[i].mse_loss(&content_layers[i], 1))
            .sum();
```
