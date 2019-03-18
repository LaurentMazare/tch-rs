#[macro_use]
extern crate failure;
extern crate tch;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    let filename = match args.as_slice() {
        [_, f] => f.to_owned(),
        _ => bail!("usage: main file.npy"),
    };

    let tensor = tch::Tensor::read_npy(filename)?;
    println!("loaded: {:?}", tensor);

    Ok(())
}
