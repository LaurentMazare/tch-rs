/* Translation with a Sequence to Sequence Model and Attention.

   This follows the line of the PyTorch tutorial:
   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
   And trains a Sequence to Sequence (seq2seq) model using attention to
   perform translation between French and English.

   The dataset can be downloaded from the following link:
   https://download.pytorch.org/tutorial/data.zip
   The eng-fra.txt file should be moved in the data directory.
*/

#[macro_use]
extern crate failure;
extern crate rand;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
extern crate tch;
use tch::nn::{GRUState, Module, OptimizerConfig, RNN};
use tch::{nn, Device, Kind, Tensor};

mod dataset;
use dataset::Dataset;
mod lang;

const MAX_LENGTH: usize = 10;
const LEARNING_RATE: f64 = 0.0001;
const SAMPLES: usize = 100_000;
const HIDDEN_SIZE: usize = 256;

struct Encoder {
    embedding: nn::Embedding,
    gru: nn::GRU,
}

impl Encoder {
    fn new(vs: nn::Path, in_dim: usize, hidden_dim: usize) -> Self {
        let gru = nn::gru(&vs, in_dim as i64, hidden_dim as i64, Default::default());
        let embedding = nn::embedding(&vs, in_dim as i64, hidden_dim as i64, Default::default());
        Encoder { embedding, gru }
    }

    fn forward(&self, xs: &Tensor, state: &GRUState) -> (Tensor, GRUState) {
        let xs = self.embedding.forward(&xs).view(&[1, -1]);
        let state = self.gru.step(&xs, &state);
        (state.value(), state)
    }

    fn zero_state(&self) -> GRUState {
        self.gru.zero_state(1)
    }
}

pub fn main() -> failure::Fallible<()> {
    let dataset = Dataset::new("eng", "fra", MAX_LENGTH)?.reverse();
    let ilang = dataset.input_lang();
    let olang = dataset.output_lang();
    let pairs = dataset.pairs();
    println!("Input:  {} {} words.", ilang.name(), ilang.len());
    println!("Output: {} {} words.", olang.name(), olang.len());
    println!("Pairs:  {}.", pairs.len());
    let mut rng = thread_rng();
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let encoder = Encoder::new(vs.root(), ilang.len(), HIDDEN_SIZE);
    let opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;
    for idx in 1..=SAMPLES {
        let (input_, target) = pairs.choose(&mut rng).unwrap();
    }
    Ok(())
}
