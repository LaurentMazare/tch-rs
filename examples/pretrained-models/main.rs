// This example illustrates how to use a pre-trained ResNet-18 model to get
// the imagenet label for some image.
// The resnet18.ot file containing the pre-trained weights can be found here:
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot
#[macro_use]
extern crate failure;
extern crate tch;
use tch::nn::ModuleT;
use tch::vision::imagenet;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    let (weight_file, image_file) = match args.as_slice() {
        [_, w, i] => (w.to_owned(), i.to_owned()),
        _ => bail!("usage: main resnet18.ot image.jpg"),
    };
    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = imagenet::load_image_and_resize(image_file)?;

    // Create a ResNet-18 model and load the weights from the file.
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let resnet18 = tch::vision::resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
    vs.load(weight_file)?;

    // Apply the forward pass of the model to get the logits.
    let output = resnet18
        .forward_t(&image.unsqueeze(0), /*train=*/ false)
        .softmax(-1); // Convert to probability.

    // Print the top 5 categories for this image.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    Ok(())
}
