extern crate tch;
use tch::{kind, Tensor};

fn main() {
    let slice = vec![0; 1_000_000];
    for i in 1..1_000_000 {
        let t = Tensor::of_slice(&slice);
        println!("{} {:?}", i, t.size())
    }
}
