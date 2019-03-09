use tch::{nn, Device, Kind, Tensor};

#[test]
fn save_and_load_var_store() {
    let filename = std::env::temp_dir().join(format!("tch-vs-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let vs1 = nn::VarStore::new(Device::Cpu);
    let vs2 = nn::VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64::from(&u1.mean()), 42.0);
    assert_eq!(f64::from(&v1.mean()), 2.0);
    assert_eq!(f64::from(&u2.mean()), 0.0);
    assert_eq!(f64::from(&v2.mean()), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64::from(&u1.mean()), 42.0);
    assert_eq!(f64::from(&u2.mean()), 42.0);
    assert_eq!(f64::from(&v2.mean()), 2.0);
}

#[test]
fn optimizer_test() {
    tch::manual_seed(42);
    // Create some linear data.
    let xs = Tensor::int_vec(&(1..15).collect::<Vec<_>>())
        .to_kind(Kind::Float)
        .view(&[-1, 1]);
    let ys = &xs * 0.42 + 1.337;

    // Fit a linear model on the data.
    let vs = nn::VarStore::new(Device::Cpu);
    let linear = nn::Linear::new(vs.root(), 1, 1);
    let opt = nn::Optimizer::sgd(&vs, 1e-3, Default::default()).unwrap();

    let loss = || {
        let predicted_ys = xs.apply(&linear);
        ((&ys - &predicted_ys) * (&ys - &predicted_ys)).mean()
    };
    let initial_loss = f64::from(loss());
    assert!(initial_loss > 1.0, "{}", initial_loss);

    // Optimization loop.
    for _idx in 1..50 {
        let loss = loss();
        opt.backward_step(&loss);
    }
    let final_loss = f64::from(loss());
    assert!(final_loss < 0.1, "{}", final_loss)
}
