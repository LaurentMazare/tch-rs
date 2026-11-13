use tch::{CModule, Device, IValue};

#[test]
fn scripted_device_results_preserve_identity_and_recover_after_errors(
) -> Result<(), Box<dyn std::error::Error>> {
    let module = CModule::load("tests/device.pt")?;
    for (name, expected) in [
        ("cpu", Device::Cpu),
        ("cuda:0", Device::Cuda(0)),
        ("cuda:2", Device::Cuda(2)),
        ("mps", Device::Mps),
        ("vulkan", Device::Vulkan),
    ] {
        assert_eq!(module.forward_is(&[IValue::String(name.into())])?, IValue::Device(expected));
    }
    for _ in 0..16 {
        let result = module.forward_is(&[IValue::String("cuda".into())]);
        assert!(result.is_err(), "unindexed CUDA must not silently become CPU: {result:?}");
        assert!(result.unwrap_err().to_string().contains("CUDA device index is unspecified"));
        assert_eq!(
            module.forward_is(&[IValue::String("cpu".into())])?,
            IValue::Device(Device::Cpu)
        );
    }
    Ok(())
}
