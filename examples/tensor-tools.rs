// A small tensor tool utility.
//
// - List the content of some npy/npz/ot file.
//     tensor-tools ls a.npy b.npz c.ot
//
// - Convert npz/ot files.
//     tensor-tools cp src.npz dst.ot
//   or
//     tensor-tools cp src.ot dst.npz

#[macro_use]
extern crate failure;
extern crate tch;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() >= 2,
        "usage: {} (ls|npz-to-ot|ot-to-npz) ...",
        args[0]
    );
    match args[1].as_str() {
        "ls" => {
            for filename in args.iter().skip(2) {
                if filename.ends_with(".npy") {
                    let tensor = tch::Tensor::read_npy(filename)?;
                    println!("{}: {:?}", filename, tensor);
                } else if filename.ends_with(".npz") {
                    let tensors = tch::Tensor::read_npz(filename)?;
                    for (name, tensor) in tensors.iter() {
                        println!("{}: {} {:?}", filename, name, tensor)
                    }
                } else if filename.ends_with(".ot") {
                    let tensors = tch::Tensor::load_multi(filename)?;
                    for (name, tensor) in tensors.iter() {
                        println!("{}: {} {:?}", filename, name, tensor)
                    }
                } else {
                    bail!("unhandled file {}", filename);
                }
            }
        }
        "cp" => {
            ensure!(args.len() == 4, "usage: {} cp src.ot dst.npz", args[0]);
            let src_filename = &args[2];
            let dst_filename = &args[3];
            let tensors = if src_filename.ends_with(".npz") {
                tch::Tensor::read_npz(src_filename)?
            } else if src_filename.ends_with(".ot") {
                tch::Tensor::load_multi(src_filename)?
            } else {
                bail!("unhandled file {}", src_filename)
            };
            for (name, tensor) in tensors.iter() {
                println!("{}: {} {:?}", src_filename, name, tensor)
            }
            if dst_filename.ends_with(".npz") {
                tch::Tensor::write_npz(&tensors, dst_filename)?
            } else if dst_filename.ends_with(".npz") {
                tch::Tensor::save_multi(&tensors, dst_filename)?
            } else {
                bail!("unhandled file {}", dst_filename)
            };
        }
        _ => bail!("usage: {} (ls|npz-to-ot|ot-to-npz) ...", args[0]),
    }

    Ok(())
}
