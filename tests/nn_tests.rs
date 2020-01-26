use tch::nn::layer_norm;
use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Kind, Reduction, Tensor};

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
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
    };
    let mut linear = nn::linear(vs.root(), 1, 1, cfg);

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

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

#[test]
fn gradient_descent_test() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys).pow(2).sum(Kind::Float);
        opt.backward_step(&loss);
    }
}

#[test]
fn bn_test() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let bn = nn::batch_norm1d(vs.root(), 40, Default::default());
    let x = Tensor::randn(&[10, 40], opts);
    let _y = x.apply_t(&bn, true);
}

#[test]
fn layer_norm_test() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let ln = layer_norm(vs.root(), vec![5, 10, 10], Default::default());
    let x = Tensor::randn(&[20, 5, 10, 10], opts);
    let _y = x.apply(&ln);
}

#[test]
fn layer_norm_parameters_test() {
    tch::manual_seed(42);
    // Create some linear data.
    let xs = Tensor::of_slice(&[42.0, 42.0, 42.0, 24.0])
        .to_kind(Kind::Float)
        .view([-1, 2]);
    let ys = &xs * 0.42 + 1.337;

    // Fit a layer normalization layer (with deterministic initialization) on the data.
    let vs = nn::VarStore::new(Device::Cpu);
    let mut opt = nn::Sgd::default().build(&vs, 1.0).unwrap();
    let mut ln = layer_norm(vs.root(), vec![2], Default::default());

    let loss = xs.apply(&ln).mse_loss(&ys, Reduction::Mean);
    let initial_loss = f64::from(&loss);
    assert!(initial_loss > 1.0, "initial loss {}", initial_loss);

    // Optimization loop.
    for _idx in 1..50 {
        let loss = xs.apply(&ln).mse_loss(&ys, Reduction::Mean);
        opt.backward_step(&loss);
    }
    let loss = xs.apply(&ln).mse_loss(&ys, Reduction::Mean);
    let final_loss = f64::from(loss);
    assert!(final_loss < 0.25, "final loss {:?}", final_loss);

    //     Reset the weights to their initial values.
    tch::no_grad(|| {
        if let Some(ws) = &mut ln.ws {
            ws.init(nn::Init::Const(1.));
        }
        if let Some(bs) = &mut ln.bs {
            bs.init(nn::Init::Const(0.));
        }
    });
    let initial_loss2 = f64::from(xs.apply(&ln).mse_loss(&ys, Reduction::Mean));
    assert_eq!(initial_loss, initial_loss2)
}

fn gru_test(rnn_config: nn::RNNConfig) {
    use nn::RNN;
    let batch_dim = 5;
    let seq_len = 3;
    let input_dim = 2;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let gru = nn::gru(&vs.root(), input_dim, output_dim, rnn_config);

    let num_directions = if rnn_config.bidirectional { 2 } else { 1 };
    let layer_dim = rnn_config.num_layers * num_directions;
    //
    // step test
    let input = Tensor::randn(&[batch_dim, input_dim], kind::FLOAT_CPU);
    let nn::GRUState(output) = gru.step(&input, &gru.zero_state(batch_dim));
    assert_eq!(output.size(), [layer_dim, batch_dim, output_dim]);

    // seq test
    let input = Tensor::randn(&[batch_dim, seq_len, input_dim], kind::FLOAT_CPU);
    let (output, _) = gru.seq(&input);
    assert_eq!(
        output.size(),
        [batch_dim, seq_len, output_dim * num_directions]
    );
}

#[test]
fn gru() {
    gru_test(Default::default());
    gru_test(nn::RNNConfig {
        bidirectional: true,
        ..Default::default()
    });
    gru_test(nn::RNNConfig {
        num_layers: 2,
        ..Default::default()
    });
    gru_test(nn::RNNConfig {
        num_layers: 2,
        bidirectional: true,
        ..Default::default()
    });
}

fn lstm_test(rnn_config: nn::RNNConfig) {
    use nn::RNN;
    let batch_dim = 5;
    let seq_len = 3;
    let input_dim = 2;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let lstm = nn::lstm(&vs.root(), input_dim, output_dim, rnn_config);

    let num_directions = if rnn_config.bidirectional { 2 } else { 1 };
    let layer_dim = rnn_config.num_layers * num_directions;
    //
    // step test
    let input = Tensor::randn(&[batch_dim, input_dim], kind::FLOAT_CPU);
    let nn::LSTMState((h, c)) = lstm.step(&input, &lstm.zero_state(batch_dim));
    assert_eq!(h.size(), [layer_dim, batch_dim, output_dim]);
    assert_eq!(c.size(), [layer_dim, batch_dim, output_dim]);

    // seq test
    let input = Tensor::randn(&[batch_dim, seq_len, input_dim], kind::FLOAT_CPU);
    let (output, _) = lstm.seq(&input);
    assert_eq!(
        output.size(),
        [batch_dim, seq_len, output_dim * num_directions]
    );
}

#[test]
fn lstm() {
    lstm_test(Default::default());
    lstm_test(nn::RNNConfig {
        bidirectional: true,
        ..Default::default()
    });
    lstm_test(nn::RNNConfig {
        num_layers: 2,
        ..Default::default()
    });
    lstm_test(nn::RNNConfig {
        num_layers: 2,
        bidirectional: true,
        ..Default::default()
    });
}
