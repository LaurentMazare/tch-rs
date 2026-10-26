// The random number generator used to initialize variables is global to the process
// and the tests within a file run in parallel, so this file must only contain a single
// test for the seeded initializations to be deterministic.
use tch::{nn, Device};

#[test]
fn weights_do_not_depend_on_bias() {
    // As in PyTorch, the weights are created before the bias so that for a given
    // seed they are the same whether the layer has a bias or not.
    let vs = nn::VarStore::new(Device::Cpu);

    let linear = |name: &str, bias: bool| {
        tch::manual_seed(42);
        nn::linear(vs.root().sub(name), 10, 4, nn::LinearConfig { bias, ..Default::default() })
    };
    assert!(linear("linear_with_bias", true).ws.equal(&linear("linear_without_bias", false).ws));

    // The default bias init for convolutions is constant, use a random one so that
    // it consumes some of the random number generator state.
    let bs_init = nn::Init::Uniform { lo: -1., up: 1. };

    let conv = |name: &str, bias: bool| {
        tch::manual_seed(42);
        let cfg = nn::ConvConfig { bias, bs_init, ..Default::default() };
        nn::conv2d(vs.root().sub(name), 2, 4, 3, cfg)
    };
    assert!(conv("conv_with_bias", true).ws.equal(&conv("conv_without_bias", false).ws));

    let conv_transpose = |name: &str, bias: bool| {
        tch::manual_seed(42);
        let cfg = nn::ConvTransposeConfig { bias, bs_init, ..Default::default() };
        nn::conv_transpose2d(vs.root().sub(name), 2, 4, 3, cfg)
    };
    assert!(conv_transpose("convt_with_bias", true)
        .ws
        .equal(&conv_transpose("convt_without_bias", false).ws));
}
