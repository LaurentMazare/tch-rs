use tch::{vision::image, TchError, Tensor};

fn check_encoder(extension: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        std::env::temp_dir().join(format!("tch-image-save-{}-{extension}", std::process::id()));
    std::fs::create_dir(&root)?;
    let pixels = Tensor::from_slice(&[10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        .reshape([3, 2, 2]);
    let output = root.join(format!("output.{extension}"));
    image::save(&pixels, &output)?;
    let decoded = image::load(&output)?;
    assert_eq!(decoded.size(), pixels.size());
    if extension != "jpg" {
        assert!(decoded.equal(&pixels));
    }
    // Existing singleton-batch input support still reaches the same encoder.
    image::save(&pixels.unsqueeze(0), &output)?;
    assert_eq!(image::load(&output)?.size(), pixels.size());

    let missing_parent = root.join("missing").join(format!("output.{extension}"));
    let missing_result = image::save(&pixels, &missing_parent);
    let directory_target = root.join(format!("directory.{extension}"));
    std::fs::create_dir(&directory_target)?;
    let directory_result = image::save(&pixels, &directory_target);
    // Check error-state recovery using the real encoder, not a stubbed status.
    image::save(&pixels, &output)?;
    assert_eq!(image::load(&output)?.size(), pixels.size());
    std::fs::remove_file(output)?;
    std::fs::remove_dir(directory_target)?;
    std::fs::remove_dir(root)?;
    assert!(
        matches!(missing_result, Err(TchError::Io(_))),
        "missing output parent must return an I/O error: {missing_result:?}"
    );
    assert!(
        matches!(directory_result, Err(TchError::Io(_))),
        "directory output must return an I/O error: {directory_result:?}"
    );
    Ok(())
}

#[test]
fn png_write_success_and_failures() -> Result<(), Box<dyn std::error::Error>> {
    check_encoder("png")
}
#[test]
fn jpeg_write_success_and_failures() -> Result<(), Box<dyn std::error::Error>> {
    check_encoder("jpg")
}
#[test]
fn bmp_write_success_and_failures() -> Result<(), Box<dyn std::error::Error>> {
    check_encoder("bmp")
}
#[test]
fn tga_write_success_and_failures() -> Result<(), Box<dyn std::error::Error>> {
    check_encoder("tga")
}
