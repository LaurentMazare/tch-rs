use std::convert::TryFrom;

use image::{
    DynamicImage, EncodableLayout, GrayAlphaImage, GrayImage, Rgb32FImage, RgbImage, Rgba32FImage,
    RgbaImage,
};

use crate::vision::image::{chw_to_hwc, hwc_to_chw};
use crate::{Kind, TchError, Tensor};

impl<'i> From<&'i DynamicImage> for Tensor {
    fn from(image: &'i DynamicImage) -> Self {
        match image {
            DynamicImage::ImageLuma8(gray) => Tensor::from(gray),
            DynamicImage::ImageLumaA8(gray_a) => Tensor::from(gray_a),
            DynamicImage::ImageRgb8(rgb) => Tensor::from(rgb),
            DynamicImage::ImageRgba8(rgba) => Tensor::from(rgba),
            DynamicImage::ImageLuma16(_) => unimplemented!(),
            DynamicImage::ImageLumaA16(_) => unimplemented!(),
            DynamicImage::ImageRgb16(_) => unimplemented!(),
            DynamicImage::ImageRgba16(_) => unimplemented!(),
            DynamicImage::ImageRgb32F(rgb) => Tensor::from(rgb),
            DynamicImage::ImageRgba32F(rgba) => Tensor::from(rgba),
            _ => {
                unimplemented!()
            }
        }
    }
}

impl<'i> From<&'i GrayImage> for Tensor {
    ///  `h * w` => `1 * 1 * h * w`
    fn from(gray: &'i GrayImage) -> Self {
        let kind = Kind::Uint8;
        let size = &[gray.height() as i64, gray.width() as i64, 1];
        let tensor =
            Tensor::f_of_data_size(gray.as_bytes(), size, kind).expect("Failed to create tensor");
        hwc_to_chw(&tensor)
    }
}

impl<'i> From<&'i GrayAlphaImage> for Tensor {
    ///  `2 * h * w` => `1 * 1 * h * w`
    fn from(gray: &'i GrayAlphaImage) -> Self {
        let kind = Kind::Uint8;
        let size = &[gray.height() as i64, gray.width() as i64, 2];
        let tensor =
            Tensor::f_of_data_size(gray.as_bytes(), size, kind).expect("Failed to create tensor");
        hwc_to_chw(&tensor)
    }
}

impl<'i> From<&'i RgbImage> for Tensor {
    /// `h * w * 3` => `1 * 3 * h * w`
    fn from(rgb: &'i RgbImage) -> Self {
        let kind = Kind::Uint8;
        let size = &[rgb.height() as i64, rgb.width() as i64, 3];
        let tensor =
            Tensor::f_of_data_size(rgb.as_raw(), size, kind).expect("Failed to create tensor");
        hwc_to_chw(&tensor)
    }
}

impl<'i> From<&'i RgbaImage> for Tensor {
    /// `h * w * 4` => `1 * 4 * h * w`
    fn from(rgb: &'i RgbaImage) -> Self {
        let kind = Kind::Uint8;
        let size = &[rgb.height() as i64, rgb.width() as i64, 4];
        let tensor =
            Tensor::f_of_data_size(rgb.as_raw(), size, kind).expect("Failed to create tensor");
        hwc_to_chw(&tensor)
    }
}

impl<'i> TryFrom<&'i Tensor> for RgbImage {
    type Error = TchError;
    ///  `1 * 3 * h * w` => `h * w * 3`
    fn try_from(value: &'i Tensor) -> Result<Self, Self::Error> {
        let tensor = assert_tensor_as_image(value, Kind::Uint8, 3)?;
        let width = tensor.size()[1] as u32;
        let height = tensor.size()[0] as u32;
        let length = (width * height * 3) as usize;
        let mut buffer = vec![0; length];
        tensor.f_copy_data(&mut buffer, length)?;
        match RgbImage::from_raw(width, height, buffer) {
            Some(s) => Ok(s),
            None => Err(TchError::Convert("Failed to convert tensor to image".to_string())),
        }
    }
}

impl<'i> From<&'i Rgb32FImage> for Tensor {
    /// `h * w * 3` => `1 * 3 * h * w`
    fn from(rgb: &'i Rgb32FImage) -> Self {
        let kind = Kind::Float;
        let size = &[rgb.height() as i64, rgb.width() as i64, 3];
        let tensor =
            Tensor::f_of_data_size(rgb.as_bytes(), size, kind).expect("Failed to create tensor");
        hwc_to_chw(&tensor)
    }
}

impl<'i> From<&'i Rgba32FImage> for Tensor {
    /// `h * w * 4` => `1 * 4 * h * w`
    fn from(rgb: &'i Rgba32FImage) -> Self {
        let kind = Kind::Float;
        let size = &[rgb.height() as i64, rgb.width() as i64, 4];
        let tensor =
            Tensor::f_of_data_size(rgb.as_bytes(), size, kind).expect("Failed to create tensor");
        hwc_to_chw(&tensor)
    }
}

impl<'i> TryFrom<&'i Tensor> for Rgb32FImage {
    type Error = TchError;
    ///  `1 * 3 * h * w` => `h * w * 3`
    fn try_from(value: &'i Tensor) -> Result<Self, Self::Error> {
        let tensor = assert_tensor_as_image(value, Kind::Float, 3)?;
        let width = tensor.size()[1] as u32;
        let height = tensor.size()[0] as u32;
        let length = (width * height * 3) as usize;
        let mut buffer = vec![0.0; length];
        tensor.f_copy_data(&mut buffer, length)?;
        match Rgb32FImage::from_raw(width, height, buffer) {
            Some(s) => Ok(s),
            None => Err(TchError::Convert("Failed to convert tensor to image".to_string())),
        }
    }
}

#[inline]
fn assert_tensor_as_image(tensor: &Tensor, except: Kind, channel: i64) -> Result<Tensor, TchError> {
    let kind = tensor.kind();
    let size = tensor.size();
    if size.len() != 3 {
        let msg = format!("Tensor size is `{size:?}`, except rank 3 tensor", size = size);
        Err(TchError::Convert(msg))?;
    }
    if size[0] != channel {
        let msg = format!(
            "Tensor size is `{size:?}`, except {channel} channels",
            size = size,
            channel = channel
        );
        Err(TchError::Convert(msg))?;
    }
    if kind != except {
        let msg = format!(
            "Tensor kind is `{kind:?}`, except `{except:?}` tensor",
            kind = kind,
            except = except
        );
        Err(TchError::Convert(msg))?;
    }
    Ok(chw_to_hwc(tensor))
}

// #[test]
// fn dump_u8() {
//     use crate::vision::image::save;
//     use image::io::Reader;
//     let img = Reader::open("img_in.png").unwrap().decode().unwrap();
//     let tensor = Tensor::from(&img);
//     save(&tensor, "img_tensor.png").unwrap();
//     let img2 = RgbImage::try_from(&tensor).unwrap();
//     img2.save("img_out.png").unwrap();
// }
//
//
// #[test]
// fn dump_f32() {
//     use crate::vision::image::save;
//     use image::io::Reader;
//     let img = Reader::open("img_in.png").unwrap().decode().unwrap().to_rgb32f();
//     let tensor = Tensor::from(&img);
//     // in fact does not support f32
//     save(&tensor, "img_tensor.png").unwrap();
//     let img2 = Rgb32FImage::try_from(&tensor).unwrap();
//     DynamicImage::ImageRgb32F(img2).to_rgb8().save("img_out.png").unwrap();
// }
