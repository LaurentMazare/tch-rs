// This example illustrates how to use pre-trained vision models.
// model to get the imagenet label for some image.
//
// The pre-trained weight files containing the pre-trained weights can be found here:
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet34.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet121.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg16.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_0.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_1.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/alexnet.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/inception-v3.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/mobilenet-v2.ot
#[macro_use]
extern crate failure;
extern crate tch;
use tch::nn::ModuleT;
use tch::vision::{alexnet, densenet, imagenet, inception, mobilenet, resnet, squeezenet, vgg};

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    let (weights, image) = match args.as_slice() {
        [_, w, i] => (std::path::Path::new(w), i.to_owned()),
        _ => bail!("usage: main resnet18.ot image.jpg"),
    };
    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = imagenet::load_image_and_resize224(image)?;

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net: Box<dyn ModuleT> = match weights.file_name().unwrap().to_str().unwrap() {
        "resnet18.ot" => Box::new(resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT)),
        "resnet34.ot" => Box::new(resnet::resnet34(&vs.root(), imagenet::CLASS_COUNT)),
        "densenet121.ot" => Box::new(densenet::densenet121(&vs.root(), imagenet::CLASS_COUNT)),
        "vgg16.ot" => Box::new(vgg::vgg16(&vs.root(), imagenet::CLASS_COUNT)),
        "squeezenet1_0.ot" => Box::new(squeezenet::v1_0(&vs.root(), imagenet::CLASS_COUNT)),
        "squeezenet1_1.ot" => Box::new(squeezenet::v1_1(&vs.root(), imagenet::CLASS_COUNT)),
        "alexnet.ot" => Box::new(alexnet::alexnet(&vs.root(), imagenet::CLASS_COUNT)),
        "inception-v3.ot" => Box::new(inception::v3(&vs.root(), imagenet::CLASS_COUNT)),
        "mobilenet-v2.ot" => Box::new(mobilenet::v2(&vs.root(), imagenet::CLASS_COUNT)),
        _ => bail!("unknown model, use a weight file named e.g. resnet18.ot"),
    };
    vs.load(weights)?;

    // Apply the forward pass of the model to get the logits.
    let output = net
        .forward_t(&image.unsqueeze(0), /*train=*/ false)
        .softmax(-1); // Convert to probability.

    // Print the top 5 categories for this image.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    Ok(())
}
