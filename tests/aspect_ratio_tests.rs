use tch::{vision::image, Device, Kind, Tensor};

fn pixels(height: i64, width: i64) -> Tensor {
    Tensor::arange(3 * height * width, (Kind::Uint8, Device::Cpu)).reshape([3, height, width])
}

#[test]
fn landscape_matching_ratio_resizes_without_extra_crop() {
    let input = pixels(2, 6);
    let result = image::resize_preserve_aspect_ratio(&input, 3, 1).unwrap();
    assert_eq!(result.size(), [3, 1, 3]);
    assert!(result.equal(&image::resize(&input, 3, 1).unwrap()));
}

#[test]
fn portrait_matching_ratio_resizes_without_extra_crop() {
    let input = pixels(6, 2);
    let result = image::resize_preserve_aspect_ratio(&input, 1, 3).unwrap();
    assert_eq!(result.size(), [3, 3, 1]);
    assert!(result.equal(&image::resize(&input, 1, 3).unwrap()));
}

#[test]
fn landscape_center_crop_keeps_unscaled_pixels() {
    let input = pixels(2, 6);
    let result = image::resize_preserve_aspect_ratio(&input, 4, 2).unwrap();
    assert!(result.equal(&input.narrow(2, 1, 4)));
}

#[test]
fn portrait_center_crop_keeps_unscaled_pixels() {
    let input = pixels(6, 2);
    let result = image::resize_preserve_aspect_ratio(&input, 2, 4).unwrap();
    assert!(result.equal(&input.narrow(1, 1, 4)));
}

#[test]
fn upscale_then_crop_preserves_the_source_ratio() {
    let input = pixels(2, 6);
    let expected = image::resize(&input, 12, 4).unwrap().narrow(2, 2, 8);
    let result = image::resize_preserve_aspect_ratio(&input, 8, 4).unwrap();
    assert_eq!(result.size(), [3, 4, 8]);
    assert!(result.equal(&expected));
}

#[test]
fn square_control_retains_existing_resize_behavior() {
    let input = pixels(4, 4);
    assert!(image::resize_preserve_aspect_ratio(&input, 2, 2)
        .unwrap()
        .equal(&image::resize(&input, 2, 2).unwrap()));
}

#[test]
fn encoded_file_and_memory_consumers_use_the_same_geometry(
) -> Result<(), Box<dyn std::error::Error>> {
    let input = pixels(2, 6);
    let path = std::env::temp_dir().join(format!("tch-aspect-ratio-{}.png", std::process::id()));
    assert!(!path.exists());
    image::save(&input, &path)?;
    let encoded = std::fs::read(&path)?;
    let file_result = image::load_and_resize(&path, 3, 1);
    let memory_result = image::load_and_resize_from_memory(&encoded, 3, 1);
    std::fs::remove_file(path)?;
    let expected = image::resize(&input, 3, 1)?;
    assert!(file_result?.equal(&expected));
    assert!(memory_result?.equal(&expected));
    Ok(())
}
