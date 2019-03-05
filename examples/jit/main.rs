// This example illustrates how to use a PyTorch model trained and exported using the
// Python JIT API.
// See https://pytorch.org/tutorials/advanced/cpp_export.html for more details.
#[macro_use]
extern crate failure;
extern crate tch;
use tch::vision::imagenet;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    let (model_file, image_file) = match args.as_slice() {
        [_, m, i] => (m.to_owned(), i.to_owned()),
        _ => bail!("usage: main model.pt image.jpg"),
    };
    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = imagenet::load_image_and_resize(image_file)?;

    // Load the Python saved module.
    let model = tch::CModule::load(model_file)?;

    // Apply the forward pass of the model to get the logits.
    let output = model.forward(&[image.unsqueeze(0)])?.softmax(-1);

    // Print the top 5 categories for this image.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    Ok(())
}
