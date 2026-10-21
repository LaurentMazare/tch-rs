use std::sync::Arc;
use tch::{nn, nn::ModuleT, Device, IValue, Kind, Tensor, TrainableCModule};

#[test]
fn dropout_honors_mode_and_explicit_setters() -> Result<(), Box<dyn std::error::Error>> {
    let vs = nn::VarStore::new(Device::Cpu);
    let mut model = TrainableCModule::load("tests/mode_dropout.pt", vs.root())?;
    let x = Tensor::ones([4, 2], (Kind::Float, Device::Cpu));
    assert!(model.forward_t(&x, false).equal(&x));
    assert_eq!(f64::try_from(model.forward_t(&x, true).sum(Kind::Float))?, 0.);
    model.set_eval();
    assert!(model.forward_ts(&[&x])?.equal(&x));
    model.set_train();
    assert_eq!(f64::try_from(model.forward_ts(&[&x])?.sum(Kind::Float))?, 0.);
    assert!(model.forward_t(&x, false).equal(&x));
    model.f_set_train()?;
    let out = model.forward_is(&[IValue::Tensor(x.shallow_clone())])?;
    match out {
        IValue::Tensor(t) => assert_eq!(f64::try_from(t.sum(Kind::Float))?, 0.),
        _ => panic!("expected tensor"),
    }
    model.f_set_eval()?;
    assert!(model.forward_ts(&[&x])?.equal(&x));
    let path = std::env::temp_dir().join(format!("tch-mode-save-{}.pt", std::process::id()));
    model.save(&path)?;
    let mut saved = std::fs::File::open(&path)?;
    let restored_vs = nn::VarStore::new(Device::Cpu);
    let restored = TrainableCModule::load_data(&mut saved, restored_vs.root())?;
    assert!(restored.forward_t(&x, false).equal(&x));
    drop(saved);
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn batchnorm_updates_only_in_training() -> Result<(), Box<dyn std::error::Error>> {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = TrainableCModule::load("tests/mode_batchnorm.pt", vs.root())?;
    let x = Tensor::from_slice(&[1f32, 3., 5., 7.]).reshape([2, 2]);
    let mean = || -> Result<Tensor, tch::TchError> {
        match model.method_is::<IValue>("mean", &[])? {
            IValue::Tensor(t) => Ok(t),
            _ => panic!("fixture mean must be a tensor"),
        }
    };
    let initial = mean()?;
    let eval = model.forward_t(&x, false);
    assert!(mean()?.equal(&initial));
    let train = model.forward_t(&x, true);
    let updated = mean()?;
    assert!(!updated.equal(&initial));
    assert!(!eval.equal(&train));
    let _ = model.forward_t(&x, false);
    assert!(mean()?.equal(&updated));
    assert!(model.method_ts::<Tensor>("mean", &[])?.equal(&updated));
    assert!(model.forward_ts(&[Tensor::ones([1], (Kind::Float, Device::Cpu))]).is_err());
    assert!(model.forward_ts(&[&x]).is_ok());
    Ok(())
}

#[test]
fn shared_forward_calls_keep_each_requested_mode() {
    tch::set_num_threads(1);
    let vs = nn::VarStore::new(Device::Cpu);
    let model = Arc::new(TrainableCModule::load("tests/mode_dropout.pt", vs.root()).unwrap());
    let threads: Vec<_> = (0..8)
        .map(|worker| {
            let model = model.clone();
            std::thread::spawn(move || {
                let x = Tensor::ones([4, 2], (Kind::Float, Device::Cpu));
                for iteration in 0..1000 {
                    let train = (worker + iteration) % 2 == 0;
                    let output = model.forward_t(&x, train);
                    assert_eq!(
                        f64::try_from(output.sum(Kind::Float)).unwrap(),
                        if train { 0. } else { 8. }
                    );
                }
            })
        })
        .collect();
    let outcomes: Vec<_> = threads.into_iter().map(|thread| thread.join()).collect();
    assert!(outcomes.iter().all(Result::is_ok));
}
