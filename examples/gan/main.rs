// Realtivistic DCGAN.
// https://github.com/AlexiaJM/RelativisticGAN
//
// TODO: override the initializations if this does not converge well.
extern crate tch;
use tch::{kind, nn, Device, Kind, Scalar, Tensor};

static IMG_SIZE: i64 = 64;
static LATENT_DIM: i64 = 128;
static BATCH_SIZE: i64 = 32;
static LEARNING_RATE: f64 = 1e-4;
static BATCHES: i64 = 100000000;

fn tr2d(p: nn::Path, c_in: i64, c_out: i64, padding: i64, stride: i64) -> nn::ConvTranspose2D {
    let cfg = nn::ConvTranspose2DConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::ConvTranspose2D::new(&p, c_in, c_out, 4, cfg)
}

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, padding: i64, stride: i64) -> nn::Conv2D {
    let cfg = nn::Conv2DConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::Conv2D::new(&p, c_in, c_out, 4, cfg)
}

fn generator(p: nn::Path) -> impl nn::ModuleT {
    nn::SequentialT::new()
        .add(tr2d(&p / "tr1", LATENT_DIM, 1024, 0, 1))
        .add(nn::BatchNorm2D::new(&p / "bn1", 1024, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(tr2d(&p / "tr2", 1024, 512, 1, 2))
        .add(nn::BatchNorm2D::new(&p / "bn2", 512, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(tr2d(&p / "tr3", 512, 256, 1, 2))
        .add(nn::BatchNorm2D::new(&p / "bn3", 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(tr2d(&p / "tr4", 256, 128, 1, 2))
        .add(nn::BatchNorm2D::new(&p / "bn4", 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(tr2d(&p / "tr5", 128, 3, 1, 2))
        .add_fn(|xs| xs.tanh())
}

fn leaky_relu(xs: &Tensor) -> Tensor {
    xs.max1(&(xs * 0.2))
}

fn discriminator(p: nn::Path) -> impl nn::ModuleT {
    nn::SequentialT::new()
        .add(conv2d(&p / "conv1", 3, 128, 1, 2))
        .add_fn(leaky_relu)
        .add(conv2d(&p / "conv2", 128, 256, 1, 2))
        .add(nn::BatchNorm2D::new(&p / "bn2", 256, Default::default()))
        .add_fn(leaky_relu)
        .add(conv2d(&p / "conv3", 256, 512, 1, 2))
        .add(nn::BatchNorm2D::new(&p / "bn3", 512, Default::default()))
        .add_fn(leaky_relu)
        .add(conv2d(&p / "conv4", 512, 1024, 1, 2))
        .add(nn::BatchNorm2D::new(&p / "bn4", 1024, Default::default()))
        .add_fn(leaky_relu)
        .add(conv2d(&p / "conv5", 1024, 1, 0, 1))
}

fn rand_latent() -> Tensor {
    Tensor::rand(&[BATCH_SIZE, LATENT_DIM], kind::FLOAT_CPU)
}

pub fn main() -> failure::Fallible<()> {
    let device = Device::cuda_if_available();
    let images = Tensor::new(); // TODO: import some images.
    let train_size = images.size()[0];

    let random_batch_images = || {
        let index = Tensor::randint(train_size, &[BATCH_SIZE], kind::INT64_CPU);
        images.index_select(0, &index).to_kind(Kind::Float) / 127.5 - 1.
    };

    let mut generator_vs = nn::VarStore::new(device);
    let generator = generator(generator_vs.root());
    let opt_g = nn::optimizer::adam(&generator_vs, LEARNING_RATE, 0.5, 0.999, 0.)?;

    let mut discriminator_vs = nn::VarStore::new(device);
    let discriminator = discriminator(discriminator_vs.root());
    let opt_d = nn::optimizer::adam(&discriminator_vs, LEARNING_RATE, 0.5, 0.999, 0.)?;

    let fixed_noise = rand_latent();

    for index in 0..BATCHES {
        discriminator_vs.unfreeze();
        generator_vs.freeze();
        let discriminator_loss = {
            let batch_images = random_batch_images();
            let y_pred = batch_images.apply_t(&discriminator, true);
            let y_pred_fake = rand_latent()
                .apply_t(&generator, true)
                .copy()
                .detach()
                .apply_t(&discriminator, true);
            y_pred.mse_loss(&(y_pred_fake.mean() + 1), 1)
                + y_pred_fake.mse_loss(&(y_pred.mean() - 1), 1)
        };
        opt_d.backward_step(&discriminator_loss);

        discriminator_vs.freeze();
        generator_vs.unfreeze();

        let generator_loss = {
            let batch_images = random_batch_images();
            let y_pred = batch_images.apply_t(&discriminator, true);
            let y_pred_fake = rand_latent()
                .apply_t(&generator, true)
                .apply_t(&discriminator, true);
            y_pred.mse_loss(&(y_pred_fake.mean() - 1), 1)
                + y_pred_fake.mse_loss(&(y_pred.mean() + 1), 1)
        };
        opt_g.backward_step(&generator_loss);

        if index % 10000 == 0 {
            let xs = fixed_noise
                .apply_t(&generator, true)
                .view(&[-1, 3, IMG_SIZE, IMG_SIZE])
                .to_device(Device::Cpu);
            let xs = ((xs + 1.) * 127.5)
                .clamp(&Scalar::float(0.), &Scalar::float(1.))
                .to_kind(Kind::Uint8);
            let mut ys: Vec<Tensor> = vec![];
            for i in 0..4 {
                ys.push(Tensor::cat(
                    &(0..4)
                        .map(|j| xs.narrow(0, 4 * i + j, 1))
                        .collect::<Vec<_>>(),
                    2,
                ))
            }
            tch::vision::image::save(&Tensor::cat(&ys, 3), format!("relout{}.png", index))?
        }
    }

    Ok(())
}
