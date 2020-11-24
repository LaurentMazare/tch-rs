// This example illustrates how to use a quantized PyTorch model trained and
// exported using the Python JIT API. Add
// See https://pytorch.org/tutorials/advanced/cpp_export.html and
// https://pytorch.org/docs/stable/quantization.html for more details.
use std::time::SystemTime;
extern crate tch;
use anyhow::{bail, Result};
use tch::vision::imagenet;

pub fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let (model_file, image_file, qengine) = match args.as_slice() {
        [_, m, i, q] => (m.to_owned(), i.to_owned(), q.to_owned()),
        [_, m, i] => (m.to_owned(), i.to_owned(), "none".to_string()),
        _ => bail!("usage: main model.pt image.jpg [fbgemm | qnnpack]"),
    };

    // Set quantization engine
    match qengine.as_str() {
        "fbgemm" => tch::QEngine::FBGEMM.set()?,
        "qnnpack" => tch::QEngine::QNNPACK.set()?,
        "none" => (),
        _ => bail!("qengine should be one of 'fbgemm' or 'qnnpack' or ommitted"),
    };

    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = imagenet::load_image_and_resize224(image_file)?;

    // Load the Python saved module.
    let model = tch::CModule::load(model_file)?;

    // Time inference
    let now = SystemTime::now();
    for _ in 1..10 {
        let _output = image.unsqueeze(0).apply(&model);
    }
    println!(
        "Mean Inference Time: {} ms",
        now.elapsed().unwrap().as_millis() / 10
    );

    // Apply the forward pass of the model to get the logits.
    let output = image
        .unsqueeze(0)
        .apply(&model)
        .softmax(-1, tch::Kind::Float);

    // Print the top 5 categories for this image.
    println!("Top 5 Predictions:");
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    Ok(())
}
