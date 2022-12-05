use tch::jit;
use tch::Tensor;

fn main() {

    // load model which generate from  `trans_model.py` file
    let model = jit::CModule::load(
        "./traced_bert.pt",
    )
    .unwrap();

    // just generate a small attention_mask and input_ids
    let attention_mask = Tensor::of_slice2(&[[
        101, 2040, 2001, 3958, 27227, 1029, 102, 3958, 103, 2001, 1037, 13997, 11510, 102,
    ]]);

    let input_ids = Tensor::of_slice2(&[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]);


    // infer model by use `forward_ts` method
    let result = model.forward_ts(&[attention_mask, input_ids]).unwrap();

    // show the result
    println!("{:?}", result);
    result.slice(1, 0, 10, 1).print();

    println!("Hello, world!");
}
