use tch::{nn, nn::OptimizerConfig, Device};

fn build(kind: &str, vs: &nn::VarStore) -> nn::Optimizer {
    match kind {
        "adam" => nn::Adam::default().build(vs, 0.1).unwrap(),
        "adamw" => nn::AdamW::default().build(vs, 0.1).unwrap(),
        "rmsprop" => nn::RmsProp::default().build(vs, 0.1).unwrap(),
        "sgd" => nn::Sgd::default().build(vs, 0.1).unwrap(),
        _ => unreachable!(),
    }
}

fn check(kind: &str) {
    let vs = nn::VarStore::new(Device::Cpu);
    let untouched = vs.root().set_group(0).ones("untouched", &[]);
    let changed = vs.root().set_group(1).ones("changed", &[]);
    let mut optimizer = build(kind, &vs);
    optimizer.set_momentum_group(1, 0.5);

    let reference_vs = nn::VarStore::new(Device::Cpu);
    let reference = reference_vs.root().ones("reference", &[]);
    let mut reference_optimizer = build(kind, &reference_vs);
    reference_optimizer.set_momentum(0.5);

    let control_vs = nn::VarStore::new(Device::Cpu);
    let control = control_vs.root().ones("control", &[]);
    let mut control_optimizer = build(kind, &control_vs);
    for gradient in [1., 3., -2.] {
        optimizer.backward_step(&((&untouched + &changed) * gradient));
        reference_optimizer.backward_step(&(&reference * gradient));
        control_optimizer.backward_step(&(&control * gradient));
        assert!(
            changed.allclose(&reference, 1e-6, 1e-6, false),
            "{kind} selected group must follow the requested momentum"
        );
        assert!(
            untouched.allclose(&control, 1e-6, 1e-6, false),
            "{kind} other groups must retain their settings"
        );
    }
    assert!(
        !changed.allclose(&untouched, 1e-6, 1e-6, false),
        "{kind} test gradients must distinguish the momentum settings"
    );
}

#[test]
fn adam_group_momentum() {
    check("adam");
}
#[test]
fn adamw_group_momentum() {
    check("adamw");
}
#[test]
fn rmsprop_group_momentum() {
    check("rmsprop");
}
#[test]
fn sgd_group_momentum() {
    check("sgd");
}
