#[macro_use]
extern crate clap;
#[macro_use]
extern crate failure;
extern crate ctrlc;
extern crate indicatif;
extern crate tch;

use failure::Fallible;
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tch::{
    nn::{ModuleT, OptimizerConfig, Sgd, VarStore},
    vision::mobilenet_v3::MobileNetV3Config,
    Device, Kind, Tensor,
};

#[derive(Debug, Clone)]
enum DatasetName {
    MNIST,
    Cifar10,
}

type PreprocessorType = Box<dyn Fn(&Tensor, &Tensor) -> (Tensor, Tensor)>;

fn main() -> Fallible<()> {
    // Setup Ctrl+C handler
    let terminate = Arc::new(AtomicBool::new(false));
    let terminate_clone = terminate.clone();
    ctrlc::set_handler(move || {
        terminate_clone.store(true, Ordering::SeqCst);
    })?;

    // Parse args
    let arg_yaml = load_yaml!("args.yaml");
    let arg_matches = clap::App::from_yaml(arg_yaml).get_matches();

    let dataset_name = match arg_matches.value_of("dataset_name") {
        Some(name) => match name {
            "mnist" => DatasetName::MNIST,
            "cifar-10" => DatasetName::Cifar10,
            _ => bail!(
                "Expect dataset name to be \"mnist\" or \"cifar-10\", but get {:?}",
                name
            ),
        },
        None => bail!("dataset name is not specified"),
    };
    let dataset_dir = match arg_matches.value_of("dataset_dir") {
        Some(path) => PathBuf::from(path),
        None => bail!("dataset directory is not specified"),
    };
    let model_file = match arg_matches.value_of("model_file") {
        Some(path) => Some(PathBuf::from(path)),
        None => None,
    };
    let epochs = match arg_matches.value_of("epochs") {
        Some(epochs) => epochs.parse()?,
        None => {
            eprintln!("warning: --epochs is not specified, defaults to 10");
            10
        }
    };
    let batch_size: i64 = match arg_matches.value_of("batch_size") {
        Some(bsize) => bsize.parse()?,
        None => {
            eprintln!("warning: --batch-size is not specified, defaults to 32");
            128
        }
    };
    let dropout = match arg_matches.value_of("dropout") {
        Some(dropout) => dropout.parse()?,
        None => {
            eprintln!("warning: --dropout is not specified, defaults to 0.8");
            0.8
        }
    };
    let width_mult = match arg_matches.value_of("width_mult") {
        Some(width_mult) => width_mult.parse()?,
        None => {
            eprintln!("warning: --width-mult is not specified, defaults to 1.0");
            1.0
        }
    };

    // Load dataset
    eprintln!("Load dataset from {:?}", &dataset_dir);

    let cifar10_mean = Tensor::of_slice(&[0.4914, 0.4822, 0.4465])
        .to_kind(Kind::Float)
        .to_device(Device::cuda_if_available())
        .view([3, 1, 1]);

    let cifar10_std = Tensor::of_slice(&[0.2023, 0.1994, 0.2010])
        .to_kind(Kind::Float)
        .to_device(Device::cuda_if_available())
        .view([3, 1, 1]);

    let (dataset, input_size, input_channel, n_classes, preprocessor) = match dataset_name {
        DatasetName::MNIST => {
            let dataset = tch::vision::mnist::load_dir(&dataset_dir)?;
            let input_size: i64 = 28;
            let input_channel: i64 = 1;
            let n_classes = 10;
            let preprocessor: PreprocessorType = Box::new(move |images, labels| {
                let images = images.view([-1, input_channel, input_size, input_size]);
                // let images = tch::vision::dataset::augmentation(&images, true, 4, 8);
                (images, labels.shallow_clone())
            });
            (dataset, input_size, input_channel, n_classes, preprocessor)
        }
        DatasetName::Cifar10 => {
            let dataset = tch::vision::cifar::load_dir(&dataset_dir)?;
            let input_size: i64 = 32;
            let input_channel: i64 = 3;
            let n_classes = 10;
            let preprocessor: PreprocessorType = Box::new(move |images, labels| {
                let normalized_images = (images - &cifar10_mean) / &cifar10_std;
                (normalized_images, labels.shallow_clone())
            });
            (dataset, input_size, input_channel, n_classes, preprocessor)
        }
    };
    let n_train_samples = dataset.train_images.size()[0];
    let n_test_samples = dataset.test_images.size()[0];
    eprint!(
        "Parameter summary:
- Trainset size = {}
- Testset size = {}
- Batch size = {}
- Number of classes = {}
- Input shape (HWC) = {:?}
- Total epochs = {}
- Dropout = {}
- Width multiplier = {}
",
        n_train_samples,
        n_test_samples,
        batch_size,
        n_classes,
        (input_size, input_size, input_channel),
        epochs,
        dropout,
        width_mult,
    );

    // Initialize model
    let mut vs = VarStore::new(Device::cuda_if_available());
    let root = vs.root();
    let model = tch::vision::mobilenet_v3::v3_large(
        &root,
        input_channel,
        n_classes,
        MobileNetV3Config {
            dropout,
            width_mult,
        },
    );

    // Initialize optimizer
    let mut lr = 0.1;
    let mut opt = Sgd {
        momentum: 0.9,
        wd: 4e-5,
        nesterov: true,
        ..Default::default()
    }
    .build(&vs, lr)?;

    // Try to load model parameters
    if let Some(path) = &model_file {
        if path.is_file() {
            eprintln!("Load model parameters from {:?}", path);
            vs.load(path)?;
        }
    }

    // Define epoch function
    let mut epoch_fn = |epoch, step: &mut i64, iter, n_samples, train| {
        // Update learning rate
        if train {
            lr = learning_rate_schedule(epoch);
            opt.set_lr(lr);
        }

        // Setup progress bar
        let n_batches = (n_samples + batch_size - 1) / batch_size;
        let pb = ProgressBar::new(n_batches as u64);
        let pb_style = ProgressStyle::default_bar().tick_chars("◐◓◑◒ ");
        let pb_style = if train {
            pb_style.template("{spinner:.yellow} [train {elapsed_precise} / {eta}] [{wide_bar}] {pos:3}/{len:3} {msg}")
        } else {
            pb_style.template("{spinner:.yellow} [test  {elapsed_precise} / {eta}] [{wide_bar}] {pos:3}/{len:3} {msg}")
        };
        pb.set_style(pb_style);
        pb.enable_steady_tick(500);

        let mut total_accuracy = 0.;
        let mut total_loss = 0.;

        for (images, labels) in iter {
            if terminate.load(Ordering::SeqCst) {
                return;
            }

            // Forward and backward pass on training batch
            let (loss, accuracy) = {
                let (images, labels) = preprocessor(&images, &labels);
                let bsize = images.size()[0];
                let images = tch::vision::dataset::augmentation(&images, true, 4, 8);
                let logits = model.forward_t(&images, true);
                let loss_tensor = logits.cross_entropy_for_logits(&labels);
                let loss = f64::from(&loss_tensor);
                let accuracy = f64::from(logits.accuracy_for_logits(&labels));
                total_loss += loss * bsize as f64;
                total_accuracy += accuracy * bsize as f64;

                if train {
                    opt.backward_step(&loss_tensor);
                }

                (loss, accuracy)
            };

            if train {
                *step += 1;
            }
            pb.set_message(&format!(
                "epoch: {: <4} step: {: <6} loss: {: <.5} accuracy: {: <3.2}% lr: {: <.5}",
                epoch + 1,
                *step,
                loss,
                accuracy * 100.,
                lr,
            ));
            pb.inc(1);
        }

        let mean_loss = total_loss / n_samples as f64;
        let mean_accuracy = total_accuracy / n_samples as f64;
        pb.set_message(&format!(
            "epoch: {: <4} step: {: <6} loss: {: <.5} accuracy: {: <3.2}% lr: {: <.5}",
            epoch + 1,
            *step,
            mean_loss,
            mean_accuracy * 100.,
            lr,
        ));
        pb.finish();
    };

    // Train model
    let mut step = 0;

    for epoch in 0..epochs {
        if terminate.load(Ordering::SeqCst) {
            break;
        }

        // Training steps
        let mut train_iter = dataset.train_iter(batch_size);
        train_iter
            .return_smaller_last_batch()
            .shuffle()
            .to_device(vs.device());

        epoch_fn(epoch, &mut step, train_iter, n_train_samples, true);

        if terminate.load(Ordering::SeqCst) {
            break;
        }

        // Predict on test dataset
        let mut test_iter = dataset.test_iter(batch_size);
        test_iter
            .return_smaller_last_batch()
            .shuffle()
            .to_device(vs.device());

        epoch_fn(epoch, &mut step, test_iter, n_test_samples, false);
    }

    // Store model parameters
    if let Some(path) = &model_file {
        eprintln!("store model parameters to {:?}", path);
        vs.save(path)?;
    }

    Ok(())
}

fn learning_rate_schedule(epoch: i64) -> f64 {
    if epoch < 50 {
        0.1
    } else if epoch < 100 {
        0.01
    } else {
        0.001
    }
}
