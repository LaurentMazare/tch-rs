use tch;
use tch::{nn, vision, Tensor};

#[test]
fn mobilenet() {
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let net = vision::mobilenet::v2(&vs.root(), 1000);
    let img = Tensor::zeros(&[1, 3, 224, 224], tch::kind::FLOAT_CPU);
    let logits = img.apply_t(&net, false);
    assert_eq!(logits.size(), [1, 1000]);
}
