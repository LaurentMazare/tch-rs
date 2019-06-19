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
extern crate tch;
use tch::nn::OptimizerConfig;
use tch::{nn, Device, Kind, Tensor};

mod dataset;
use dataset::Dataset;
mod lang;
use lang::Lang;

const MAX_LENGTH: usize = 10;
const LEARNING_RATE: f64 = 0.0001;

pub fn main() -> failure::Fallible<()> {
    let dataset = Dataset::new("eng", "fra", MAX_LENGTH)?.reverse();
    let ilang = dataset.input_lang();
    let olang = dataset.output_lang();
    println!("Input:  {} {} words.", ilang.name(), ilang.len());
    println!("Output: {} {} words.", olang.name(), olang.len());
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;
    Ok(())
}
