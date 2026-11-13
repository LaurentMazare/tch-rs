use std::path::{Path, PathBuf};
use tch::{vision::image, Device, Kind, TchError, Tensor};

fn directory(case: &str) -> std::io::Result<PathBuf> {
    let root = std::env::temp_dir().join(format!("tch-image-dir-{}-{case}", std::process::id()));
    std::fs::create_dir(&root)?;
    std::fs::create_dir(root.join("nested"))?;
    Ok(root)
}

fn cleanup(root: &Path) -> std::io::Result<()> {
    for name in ["bad.png", "nested/bad.jpg", "nested/good.png", "readme.txt"] {
        let path = root.join(name);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }
    std::fs::remove_dir(root.join("nested"))?;
    std::fs::remove_dir(root)
}

#[test]
fn all_corrupt_matching_images_return_missing_image_without_panicking(
) -> Result<(), Box<dyn std::error::Error>> {
    let root = directory("corrupt")?;
    std::fs::write(root.join("bad.png"), b"not a PNG")?;
    std::fs::write(root.join("nested/bad.jpg"), b"not a JPEG")?;
    let result = std::panic::catch_unwind(|| image::load_dir(&root, 2, 2));
    cleanup(&root)?;
    assert!(
        matches!(result, Ok(Err(TchError::MissingImage(_)))),
        "all decode failures must use the Result boundary: {result:?}"
    );
    Ok(())
}

#[test]
fn mixed_directory_keeps_valid_images_and_discards_decode_failures(
) -> Result<(), Box<dyn std::error::Error>> {
    let root = directory("mixed")?;
    std::fs::write(root.join("bad.png"), b"not a PNG")?;
    std::fs::write(root.join("nested/bad.jpg"), b"not a JPEG")?;
    let pixels = Tensor::full([3, 2, 2], 17, (Kind::Uint8, Device::Cpu));
    image::save(&pixels, root.join("nested/good.png"))?;
    let result = image::load_dir(&root, 2, 2);
    cleanup(&root)?;
    let images = result?;
    assert_eq!(images.size(), [1, 3, 2, 2]);
    assert!(images.get(0).equal(&pixels));
    Ok(())
}

#[test]
fn no_matching_filenames_retains_not_found_error() -> Result<(), Box<dyn std::error::Error>> {
    let root = directory("empty")?;
    std::fs::write(root.join("readme.txt"), b"not an image filename")?;
    let result = image::load_dir(&root, 2, 2);
    cleanup(&root)?;
    match result {
        Err(TchError::Io(error)) => assert_eq!(error.kind(), std::io::ErrorKind::NotFound),
        result => panic!("no matching files must retain NotFound: {result:?}"),
    }
    Ok(())
}
