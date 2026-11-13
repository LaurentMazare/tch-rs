use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

#[test]
fn entry_uniform_matches_path_distribution() -> Result<(), Box<dyn std::error::Error>> {
    let vs = nn::VarStore::new(Device::Cpu);
    tch::manual_seed(1234);
    let from_entry = vs.root().entry("entry_uniform").or_kaiming_uniform(&[128, 128]);
    tch::manual_seed(1234);
    let from_path = vs.root().f_kaiming_uniform("path_uniform", &[128, 128])?;
    assert!(
        from_entry.equal(&from_path),
        "Entry uniform must use the same distribution as Path uniform"
    );
    assert!(f64::try_from(from_entry.abs().max())? <= (6f64 / 128.).sqrt());

    Ok(())
}

#[test]
fn no_train_entries_stay_out_of_optimizer_and_preserve_existing_entries(
) -> Result<(), Box<dyn std::error::Error>> {
    let vs = nn::VarStore::new(Device::Cpu);
    let weight = vs.root().entry("weight").or_zeros(&[2]);
    let ones = vs.root().entry("ones").or_ones_no_train(&[2]);
    let zeros = vs.root().entry("zeros").or_zeros_no_train(&[2]);
    assert!(!ones.requires_grad());
    assert!(!zeros.requires_grad());
    assert_eq!(vs.trainable_variables().len(), 1);
    let mut optimizer = nn::Sgd::default().build(&vs, 0.1)?;
    optimizer.backward_step(&(&weight + &ones + &zeros).sum(Kind::Float));
    assert!(weight.allclose(&Tensor::from_slice(&[-0.1f32, -0.1]), 1e-6, 1e-6, false));
    assert!(ones.equal(&Tensor::ones([2], (Kind::Float, Device::Cpu))));
    assert!(zeros.equal(&Tensor::zeros([2], (Kind::Float, Device::Cpu))));
    assert!(!ones.grad().defined());
    assert!(!zeros.grad().defined());

    let existing = vs.root().entry("weight").or_ones_no_train(&[2]);
    assert!(existing.equal(&weight));
    assert!(existing.requires_grad(), "an existing entry retains its original trainability");
    assert_eq!(vs.trainable_variables().len(), 1);
    let reused = vs.root().entry("ones").or_zeros(&[2]);
    assert!(reused.equal(&ones));
    assert!(!reused.requires_grad());
    assert_eq!(vs.trainable_variables().len(), 1);
    Ok(())
}
