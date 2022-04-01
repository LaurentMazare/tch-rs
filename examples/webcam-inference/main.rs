
use anyhow::{bail, Result};
use std::{time, process};
use std::collections::HashMap;

use opencv::{
	highgui,
	prelude::*,
	imgproc,
	videoio, 
    core,
};


pub fn main() -> Result<()>{
    
    let args: Vec<_> = std::env::args().collect();
    let (model_file, video_name) = match args.as_slice() {
        [_, m, i] => (m.to_owned(), i.to_owned()),
        _ => bail!("usage: main cifar10_mobilenet_v3_small.pt ../test_video.mp4"),   
    };

    // cifar 10 classes and their indexes
    let class_names = vec!["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
    let mut class_map = HashMap::new();
    for (idx, class_name) in class_names.iter().enumerate() {
        let idx = idx as i32;
        class_map.insert(idx, class_name);
    }

    // model trained on 32x32 images. (CIFAR10)
    const CIFAR_WIDTH: i32 = 32; 
    const CIFAR_HEIGHT: i32 = 32; 

    // time that a frame will stay on screen in ms
    const DELAY: i32 = 30;

    // create video stream
    let mut capture = videoio::VideoCapture::from_file(&video_name, videoio::CAP_ANY)?;
    
    println!("Inferencing on video: {}", video_name);

    // create empty window named 'frame'
    let win_name = "frame";
    highgui::named_window(win_name, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(win_name, 640, 480)?;
    
    // create empty Mat to store image data
    let mut frame = Mat::default();

    // load jit model and put it to cuda
    let mut model = tch::CModule::load(model_file)?;   
    model.set_eval(); 
    model.to(tch::Device::Cuda(0), tch::Kind::Float, false);

    let is_video_on = capture.is_opened()?;

    if !is_video_on {
        println!("Could'not open video: {}. Aborting program.", video_name);
        process::exit(0); 
    }
    else {
        loop {
            // read frame to empty mat
            capture.read(&mut frame)?;
            // resize image
            let mut resized = Mat::default();   
            imgproc::resize(&frame, &mut resized, core::Size{width: CIFAR_WIDTH, height: CIFAR_HEIGHT}, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
            // convert bgr image to rgb
            let mut rgb_resized = Mat::default();  
            imgproc::cvt_color(&resized, &mut rgb_resized, imgproc::COLOR_BGR2RGB, 0)?;    
            // get data from Mat 
            let h = resized.size()?.height;
            let w = resized.size()?.width;   
            let resized_data = resized.data_bytes_mut()?; 
            // convert bytes to tensor
            let tensor = tch::Tensor::of_data_size(resized_data, &[h as i64, w as i64, 3], tch::Kind::Uint8);  
            // normalize image tensor
            let tensor = tensor.to_kind(tch::Kind::Float) / 255;
            // carry tensor to cuda
            let tensor = tensor.to_device(tch::Device::Cuda(0)); 
            // convert (H, W, C) to (C, H, W)
            let tensor = tensor.permute(&[2, 0, 1]);
            // add batch dim (convert (C, H, W) to (N, C, H, W)) 
            let normalized_tensor = tensor.unsqueeze(0);   

            // make prediction and time it. 
            let start = time::Instant::now();
            let probabilites = model.forward_ts(&[normalized_tensor])?.softmax(-1, tch::Kind::Float);  
            let predicted_class = i32::from(probabilites.argmax(None, false));
            let probability_of_class = f32::from(probabilites.max()); 
            let duration = start.elapsed();
            println!("Predicted class: {:?}, probability of it: {:?}, prediction time: {:?}", class_map[&predicted_class], probability_of_class, duration); 

            // show image 
            highgui::imshow(win_name, &frame)?;
            let key = highgui::wait_key(DELAY)?; 
            // if button q pressed, abort.
            if key == 113 { 
                highgui::destroy_all_windows()?;
                println!("Pressed q. Aborting program.");
                break;
            }
        }
    }
  
    Ok(())
} 
