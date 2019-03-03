/* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

   It has been heavily inspired by https://github.com/karpathy/char-rnn
*/

extern crate tch;
use tch::nn::Module;
use tch::{nn, kind, Device, Tensor};

static LEARNING_RATE: f64 = 0.01;
static HIDDEN_SIZE: i64 = 256;
static SEQ_LEN: i64 = 180;
static BATCH_SIZE: i64 = 256;
static EPOCHS: i64 = 100;
static SAMPLING_LEN: i64 = 1024;

pub fn main() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let data = tch::data::TextData::new("data/input.txt").unwrap();
    let labels = data.labels();
    println!("Dataset loaded, {} labels.", labels);
    let lstm = nn::LSTM::new(&vs.root(), labels, HIDDEN_SIZE);
    let linear = nn::Linear::new(&vs.root(), HIDDEN_SIZE, labels);
    let opt = nn::Optimizer::adam(&vs, LEARNING_RATE, Default::default());
    for epoch in 1..(1+EPOCHS) {
        for batch in data.iter_shuffle(SEQ_LEN + 1, BATCH_SIZE) {
            let xs = batch.narrow(1, 0, SEQ_LEN).view(&[BATCH_SIZE, SEQ_LEN, 1]);
            let xs_onehot =
                Tensor::zeros(&[BATCH_SIZE, SEQ_LEN, labels], kind::FLOAT_CPU)
                    .scatter_(2, &xs.to_kind(kind::Kind::Int64), &Tensor::ones(&[], kind::FLOAT_CPU));
            let ys = batch.narrow(1, 1, SEQ_LEN);
            let (lstm_out, _) = lstm.seq(&xs_onehot);
            println!("{:?} {:?}", xs_onehot, lstm_out);
            let logits = linear.forward(&lstm_out);
            let loss = logits.cross_entropy_for_logits(&ys);
            opt.backward_step_clip(&loss, 0.5);
        }

        // TODO: add some sampling at the end of each epoch.
        println!("Epoch: {}", epoch);
    }
}
