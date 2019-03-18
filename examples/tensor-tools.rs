#[macro_use]
extern crate failure;
extern crate tch;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    let filename = match args.as_slice() {
        [_, f] => f.to_owned(),
        _ => bail!("usage: main file.npy"),
    };

    if filename.ends_with(".npy") {
        let tensor = tch::Tensor::read_npy(filename)?;
        println!("loaded: {:?}", tensor);
    } else if filename.ends_with(".npz") {
        let tensors = tch::Tensor::read_npz(filename)?;
        for (name, tensor) in tensors.iter() {
            println!("{}: {:?}", name, tensor)
        }
    } else {
        bail!("unhandled file {}", filename);
    }

    Ok(())
}
