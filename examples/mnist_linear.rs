/* Linear model for the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
     train-images-idx3-ubyte.gz
     train-labels-idx1-ubyte.gz
     t10k-images-idx3-ubyte.gz
     t10k-labels-idx1-ubyte.gz

   This should rearch 91.5% accuracy.
*/

extern crate tch;
use tch::{kind, no_grad, vision, Kind, Tensor};

static IMAGE_DIM: i64 = 784;
static LABELS: i64 = 10;

fn main() {
    let m = vision::mnist::load_dir(std::path::Path::new("data")).unwrap();
    println!("train-images: {:?}", m.train_images.size());
    println!("train-labels: {:?}", m.train_labels.size());
    println!("test-images: {:?}", m.test_images.size());
    println!("test-labels: {:?}", m.test_labels.size());
    let mut ws = Tensor::zeros(&[IMAGE_DIM, LABELS], &kind::FLOAT_CPU).set_requires_grad(true);
    let mut bs = Tensor::zeros(&[LABELS], &kind::FLOAT_CPU).set_requires_grad(true);
    for epoch in 1..200 {
        let logits = m.train_images.mm(&ws) + &bs;
        let loss = logits.log_softmax(-1).nll_loss(&m.train_labels);
        ws.zero_grad();
        bs.zero_grad();
        loss.backward();
        no_grad(|| {
            ws += ws.grad() * (-1);
            bs += bs.grad() * (-1);
        });
        let test_logits = m.test_images.mm(&ws) + &bs;
        let test_accuracy = test_logits
            .argmax1(-1, false)
            .eq1(&m.test_labels)
            .to_kind(Kind::Float)
            .mean()
            .double_value(&[]);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            loss.double_value(&[]),
            100. * test_accuracy
        );
    }
}
