/* Advantage Actor Critic (A2C) model.
   A2C is a synchronous variant of Asynchronous the Advantage Actor Critic (A3C)
   model introduced by DeepMind in https://arxiv.org/abs/1602.01783

   See https://blog.openai.com/baselines-acktr-a2c/ for a reference
   python implementation.
*/
use super::gym_env::{GymEnv, Step};
use tch::{nn, nn::OptimizerConfig, Tensor};

static ENV: &'static str = "SpaceInvadersNoFrameskip-v4";
static NPROCS: i64 = 16;
static NSTEPS: i64 = 5;
static NSTACK: i64 = 4;
static UPDATES: i64 = 1000000;

fn model(p: &nn::Path, input_shape: &[i64], nact: i64) -> impl nn::Module {
    let nin = input_shape.iter().fold(1, |acc, x| acc * x);
    nn::Sequential::new()
        .add(nn::Linear::new(p / "lin1", nin, 32, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::Linear::new(p / "lin2", 32, nact, Default::default()))
}

fn accumulate_rewards(steps: &[Step]) -> Vec<f64> {
    let mut rewards: Vec<f64> = steps.iter().map(|s| s.reward).collect();
    let mut acc_reward = 0f64;
    for (i, reward) in rewards.iter_mut().enumerate().rev() {
        if steps[i].is_done {
            acc_reward = 0.0;
        }
        acc_reward += *reward;
        *reward = acc_reward;
    }
    rewards
}

pub fn run() -> cpython::PyResult<()> {
    let env = GymEnv::new(ENV, Some(NPROCS))?;
    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let model = model(&vs.root(), env.observation_space(), env.action_space());
    let opt = nn::Adam::default().build(&vs, 1e-2).unwrap();

    for epoch_idx in 0..50 {
        let mut obs = env.reset()?;
        let mut steps: Vec<Step> = vec![];
        // Perform some rollouts with the current model.
        loop {
            let action = tch::no_grad(|| {
                obs.unsqueeze(0)
                    .apply(&model)
                    .softmax(1)
                    .multinomial(1, true)
            });
            let action = i64::from(action);
            let step = env.step(action)?;
            steps.push(step.copy_with_obs(&obs));
            obs = if step.is_done { env.reset()? } else { step.obs };
            if step.is_done && steps.len() > 5000 {
                break;
            }
        }
        let sum_r: f64 = steps.iter().map(|s| s.reward).sum();
        let episodes: i64 = steps.iter().map(|s| s.is_done as i64).sum();
        println!(
            "epoch: {:<3} episodes: {:<5} avg reward per episode: {:.2}",
            epoch_idx,
            episodes,
            sum_r / episodes as f64
        );

        // Train the model via policy gradient on the rollout data.
        let batch_size = steps.len() as i64;
        let actions: Vec<i64> = steps.iter().map(|s| s.action).collect();
        let actions = Tensor::int_vec(&actions).unsqueeze(1);
        let rewards = accumulate_rewards(&steps);
        let rewards = Tensor::float_vec(&rewards);
        let action_mask = Tensor::zeros(&[batch_size, 2], tch::kind::FLOAT_CPU).scatter_(
            1,
            &actions,
            &Tensor::from(1.),
        );
        let obs: Vec<Tensor> = steps.into_iter().map(|s| s.obs).collect();
        let logits = Tensor::stack(&obs, 0).apply(&model);
        let log_probs = (action_mask * logits.log_softmax(1)).sum2(&[1], false);
        let loss = -(rewards * log_probs).mean();
        opt.backward_step(&loss)
    }
    Ok(())
}
