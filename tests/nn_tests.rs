use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Kind, Reduction, Tensor};

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
    let mut vs2 = nn::VarStore::new(Device::Cpu);
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
    let xs = Tensor::of_slice(&(1..15).collect::<Vec<_>>())
        .to_kind(Kind::Float)
        .view([-1, 1]);
    let ys = &xs * 0.42 + 1.337;

    // Fit a linear model (with deterministic initialization) on the data.
    let vs = nn::VarStore::new(Device::Cpu);
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
    };
    let mut linear = nn::linear(vs.root(), 1, 1, cfg);
    let opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();

    let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
    let initial_loss = f64::from(&loss);
    assert!(initial_loss > 1.0, "initial loss {}", initial_loss);

    // Optimization loop.
    for _idx in 1..50 {
        let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
        opt.backward_step(&loss);
    }
    let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
    let final_loss = f64::from(loss);
    assert!(final_loss < 0.25, "final loss {}", final_loss);

    // Reset the weights to their initial values.
    tch::no_grad(|| {
        linear.ws.init(nn::Init::Const(0.));
        linear.bs.init(nn::Init::Const(0.));
    });
    let initial_loss2 = f64::from(xs.apply(&linear).mse_loss(&ys, Reduction::Mean));
    assert_eq!(initial_loss, initial_loss2)
}

#[test]
fn var_store_test() {
    let vs = nn::VarStore::new(Device::Cpu);
    let zeros = vs.root().zeros("t1", &[3]);
    assert_eq!(Vec::<f64>::from(&zeros), [0., 0., 0.]);
    let zeros = vs.root().var("t2", &[3], nn::Init::Const(0.));
    assert_eq!(Vec::<f64>::from(&zeros), [0., 0., 0.]);
    let ones = vs.root().var("t3", &[3], nn::Init::Const(1.));
    assert_eq!(Vec::<f64>::from(&ones), [1., 1., 1.]);
    let forty_two = vs.root().var("t4", &[2], nn::Init::Const(42.));
    assert_eq!(Vec::<f64>::from(&forty_two), [42., 42.]);
    let uniform = vs
        .root()
        .var("t5", &[100], nn::Init::Uniform { lo: 1.0, up: 2.0 });
    let uniform_min = f64::from(&uniform.min());
    let uniform_max = f64::from(&uniform.max());
    assert!(uniform_min >= 1., "min {}", uniform_min);
    assert!(uniform_max <= 2., "max {}", uniform_max);
    let uniform_std = f64::from(&uniform.std(true));
    assert!(
        uniform_std > 0.15 && uniform_std < 0.35,
        "std {}",
        uniform_std
    );
    let mut vs2 = nn::VarStore::new(Device::Cpu);
    let ones = vs2.root().ones("t1", &[3]);
    assert_eq!(Vec::<f64>::from(&ones), [1., 1., 1.]);
    vs2.copy(&vs).unwrap();
    assert_eq!(Vec::<f64>::from(&ones), [0., 0., 0.]);
}

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

#[test]
fn gradient_descent_test() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys).pow(2).sum();
        opt.backward_step(&loss);
    }
}

#[test]
fn my_test() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let bn = nn::batch_norm1d(vs.root(), 40, Default::default());

    let x = Tensor::randn(&[10, 40], opts);
    let _y = x.apply_t(&bn, true);
}
