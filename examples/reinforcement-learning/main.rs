extern crate cpython;
extern crate tch;

mod a2c;
mod gym_env;
mod policy_gradient;
mod vec_gym_env;

fn main() -> cpython::PyResult<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.iter().map(|x| x.as_str()).collect::<Vec<_>>().as_slice() {
        [_, "a2c"] => a2c::train()?,
        [_, "a2c-sample", weight_file] => a2c::sample(weight_file)?,
        [_, "pg"] => policy_gradient::run()?,
        _ => println!("usage: main pg|a2c|a2c-sample"),
    }
    Ok(())
}
