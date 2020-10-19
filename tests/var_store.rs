use std::fs;
use tch::nn::OptimizerConfig;
use tch::{nn::Init, nn::VarStore, Device, Kind, Tensor};

#[test]
fn path_components() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();
    let path = root.sub("a");
    let path = path.sub("test");
    assert_eq!(path.components().collect::<Vec<_>>(), vec!["a", "test"]);
}

#[test]
fn var_store_entry() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let t1 = root.entry("key").or_zeros(&[3, 1, 4]);
    let t2 = root.entry("key").or_zeros(&[1, 5, 9]);

    assert_eq!(t1.size(), &[3, 1, 4]);
    assert_eq!(t2.size(), &[3, 1, 4]);
}

#[test]
fn save_and_load_var_store() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-load-complete-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 2.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn save_and_load_partial_var_store() {
    let filename = std::env::temp_dir().join(format!(
        "tch-vs-partial-load-complete-{}",
        std::process::id()
    ));
    let add = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    let missing_variables = vs2.load_partial(&filename).unwrap();
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 2.0);
    assert!(missing_variables.is_empty());
    fs::remove_file(filename).unwrap();
}

#[test]
#[should_panic]
fn save_and_load_var_store_incomplete_file() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-load-incomplete-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        u
    };
    let add_partial = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let mut u1 = add(&vs1.root());
    let (u2, v2) = add_partial(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
    });
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 1.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn save_and_load_partial_var_store_incomplete_file() {
    let filename = std::env::temp_dir().join(format!(
        "tch-vs-partial-load-incomplete-{}",
        std::process::id()
    ));
    let add = |vs: &tch::nn::Path| {
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        u
    };
    let add_partial = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let mut u1 = add(&vs1.root());
    let (u2, v2) = add_partial(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
    });
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    let missing_variables = vs2.load_partial(&filename).unwrap();
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 1.0);
    assert_eq!(missing_variables, vec!(String::from("a.b.t2")));
    fs::remove_file(filename).unwrap();
}

#[test]
fn init_test() {
    let vs = VarStore::new(Device::Cpu);
    let zeros = vs.root().zeros("t1", &[3]);
    assert_eq!(Vec::<f64>::from(&zeros), [0., 0., 0.]);
    let zeros = vs.root().var("t2", &[3], Init::Const(0.));
    assert_eq!(Vec::<f64>::from(&zeros), [0., 0., 0.]);
    let ones = vs.root().var("t3", &[3], Init::Const(1.));
    assert_eq!(Vec::<f64>::from(&ones), [1., 1., 1.]);
    let ones = vs.root().var("t4", &[3], Init::Const(0.5));
    assert_eq!(Vec::<f64>::from(&ones), [0.5, 0.5, 0.5]);
    let forty_two = vs.root().var("t4", &[2], Init::Const(42.));
    assert_eq!(Vec::<f64>::from(&forty_two), [42., 42.]);
    let uniform = vs
        .root()
        .var("t5", &[100], Init::Uniform { lo: 1.0, up: 2.0 });
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
    let normal = vs.root().var(
        "normal",
        &[100],
        Init::Randn {
            mean: 0.,
            stdev: 0.02,
        },
    );
    let normal_std = f64::from(&normal.std(true));
    assert!(normal_std <= 0.03, "std {}", normal_std);
    let mut vs2 = VarStore::new(Device::Cpu);
    let ones = vs2.root().ones("t1", &[3]);
    assert_eq!(Vec::<f64>::from(&ones), [1., 1., 1.]);
    vs2.copy(&vs).unwrap();
    assert_eq!(Vec::<f64>::from(&ones), [0., 0., 0.]);
}

fn check_param_group(mut opt: tch::nn::Optimizer<tch::nn::Sgd>, foo: Tensor, bar: Tensor) {
    opt.set_lr(0.1);
    opt.set_lr_group(0, 0.);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(0.42f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "0.00");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.42");
    opt.set_lr_group(0, 0.1);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(0f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "-0.21");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.21");
    opt.set_lr_group(7, 0.);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(0.22f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "0.01");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.21");
    // The following sets the learning rate for both groups.
    opt.set_lr(0.);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(0.42f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "0.01");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.21");
    opt.set_lr(0.1);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(0.42f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "0.11");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.31");
}

#[test]
fn param_group() {
    tch::manual_seed(42);
    let vs = VarStore::new(Device::Cpu);
    let opt = tch::nn::Sgd::default().build(&vs, 1.0).unwrap();
    let root = vs.root();
    let foo = root.set_group(0).zeros("foo", &[]);
    let bar = root.set_group(7).zeros("bar", &[]);
    check_param_group(opt, foo, bar);
}

#[test]
fn save_and_load_with_group() {
    let filename = std::env::temp_dir().join(format!("tch-vs-load-grp-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.set_group(1).sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs
            .sub("a")
            .set_group(4)
            .sub("b")
            .sub("ccc")
            .ones("t123", &[3]);
        (u, v)
    };
    let vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64::from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64::from(&v2.mean(Kind::Float)), 2.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn param_group_weight_decay() {
    tch::manual_seed(42);
    let vs = VarStore::new(Device::Cpu);
    let mut opt = tch::nn::Sgd::default().build(&vs, 0.0).unwrap();
    opt.set_lr(0.1);
    let root = vs.root();
    let foo = root.set_group(0).zeros("foo", &[]);
    let bar = root.set_group(7).zeros("bar", &[]);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(1f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "0.50");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.50");
    opt.set_weight_decay(0.1);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(1f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "0.49");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.49");
    opt.set_weight_decay_group(7, 0.);
    for _idx in 1..100 {
        let loss = (&foo + &bar).mse_loss(&Tensor::from(1f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64::from(&foo)), "0.30");
    assert_eq!(format!("{:.2}", f64::from(&bar)), "0.69");
}
