/* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

   This is mostly a rust port of https://github.com/karpathy/minGPT
*/

extern crate tch;
use anyhow::Result;
use tch::data::TextData;
use tch::nn::{ModuleT, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor, IndexOp};

const LEARNING_RATE: f64 = 0.0003;
const BLOCK_SIZE: i64 = 128;
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 1024;

#[derive(Debug, Copy, Clone)]
struct Config {
    vocab_size: i64,
    n_embd: i64,
    n_head: i64,
    n_layer: i64,
    block_size: i64,
    attn_pdrop: f64,
    resid_pdrop: f64,
    embd_pdrop: f64,
}

fn causal_self_attention(p: &nn::Path, cfg: Config) -> impl ModuleT {
    let key = nn::linear(p / "key", cfg.n_embd, cfg.n_embd, Default::default());
    let query = nn::linear(p / "query", cfg.n_embd, cfg.n_embd, Default::default());
    let value = nn::linear(p / "value", cfg.n_embd, cfg.n_embd, Default::default());
    let proj = nn::linear(p / "proj", cfg.n_embd, cfg.n_embd, Default::default());
    let mask_init = Tensor::ones(&[cfg.block_size, cfg.block_size], (Kind::Float, p.device())).tril(0);
    let mask_init = mask_init.view([1, 1, cfg.block_size, cfg.block_size]);
    // let mask = p.var_copy("mask", &mask_init);
    let mask = mask_init;
    nn::func_t(move |xs, train| {
        let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
        let sizes = [sz_b, sz_t, cfg.n_head, sz_c / cfg.n_head];
        let k = xs.apply(&key).view(sizes).transpose(1, 2);
        let q = xs.apply(&query).view(sizes).transpose(1, 2);
        let v = xs.apply(&value).view(sizes).transpose(1, 2);
        let att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));
        let att = att.masked_fill(&mask.i((.., .., ..sz_t, ..sz_t)).eq(0.), std::f64::NEG_INFINITY);
        let att = att.softmax(-1, Kind::Float).dropout(cfg.attn_pdrop, train);
        let ys = att.matmul(&v).transpose(1, 2).contiguous().view([sz_b, sz_t, sz_c]);
        ys.apply(&proj).dropout(cfg.resid_pdrop, train)
    })
}

fn block(p: &nn::Path, cfg: Config) -> impl ModuleT {
    let ln1 = nn::layer_norm(p / "ln1", vec![cfg.n_embd], Default::default());
    let ln2 = nn::layer_norm(p / "ln2", vec![cfg.n_embd], Default::default());
    let attn = causal_self_attention(p, cfg);
    let lin1 = nn::linear(p / "lin1", cfg.n_embd, 4 * cfg.n_embd, Default::default());
    let lin2 = nn::linear(p / "lin2", 4 * cfg.n_embd, cfg.n_embd, Default::default());
    nn::func_t(move |xs, train| {
        let xs = xs + xs.apply(&ln1).apply_t(&attn, train);
        let ys = xs.apply(&ln2).apply(&lin1).gelu().apply(&lin2).dropout(cfg.resid_pdrop, train);
        xs + ys
    })
}

fn gpt(p: &nn::Path, cfg: Config) -> impl ModuleT {
    let tok_emb = nn::embedding(p / "tok_emb", cfg.vocab_size, cfg.n_embd, Default::default());
    let pos_emb = p.zeros("pos_emb", &[1, cfg.block_size, cfg.n_embd]);
    let ln_f = nn::layer_norm(p / "ln_f", vec![cfg.n_embd], Default::default());
    let head = nn::linear(p / "head", cfg.n_embd, cfg.vocab_size, nn::LinearConfig { bias: false, ..Default::default()});
    let mut blocks = nn::seq_t();
    for block_idx in 0..cfg.n_layer {
        blocks = blocks.add(block(&(p / block_idx), cfg));
    }
    nn::func_t(move |xs, train| {
        let (_sz_b, sz_t) = xs.size2().unwrap();
        let tok_emb = xs.apply(&tok_emb);
        let pos_emb = pos_emb.i((.., ..sz_t, ..));
        (tok_emb + pos_emb).dropout(cfg.embd_pdrop, train).apply_t(&blocks, train).apply(&ln_f).apply(&head)
    })
}

/// Generates some sample string using the GPT model.
fn sample(data: &TextData, gpt: &impl ModuleT, device: Device) -> String {
    let mut input = Tensor::zeros(&[1, BLOCK_SIZE], (Kind::Int64, device));
    let mut result = String::new();
    for _index in 0..SAMPLING_LEN {
        let logits = input.apply_t(gpt, false).i((0, -1, ..));
        let sampled_y = logits
            .softmax(-1, Kind::Float)
            .multinomial(1, true);
        let last_label = i64::from(&sampled_y);
        result.push(data.label_to_char(last_label));
        input = Tensor::cat(&[input, sampled_y.view([1, 1])], 1).narrow(1, 1, BLOCK_SIZE);
    }
    result
}

pub fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let data = TextData::new("data/input.txt")?;
    let labels = data.labels();
    println!("Dataset loaded, {} labels.", labels);
    let cfg = Config {
        vocab_size: labels,
        n_embd: 512,
        n_head: 8,
        n_layer: 8,
        block_size: BLOCK_SIZE,
        attn_pdrop: 0.1,
        resid_pdrop: 0.1,
        embd_pdrop: 0.1,
    };
    let gpt = gpt(&(&vs.root() / "gpt"), cfg);
    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;
    for epoch in 1..(1 + EPOCHS) {
        let mut sum_loss = 0.;
        let mut cnt_loss = 0.;
        for batch in data.iter_shuffle(BLOCK_SIZE + 1, BATCH_SIZE) {
            let xs = batch.narrow(1, 0, BLOCK_SIZE).to_kind(Kind::Int64).to_device(device);
            let ys = batch.narrow(1, 1, BLOCK_SIZE).to_kind(Kind::Int64).to_device(device);
            let logits = xs.apply_t(&gpt, true);
            let loss = logits
                .view([BATCH_SIZE * BLOCK_SIZE, labels])
                .cross_entropy_for_logits(&ys.view([BATCH_SIZE * BLOCK_SIZE]));
            opt.backward_step_clip(&loss, 0.5);
            sum_loss += f64::from(loss);
            cnt_loss += 1.0;
        }
        println!("Epoch: {}   loss: {:5.3}", epoch, sum_loss / cnt_loss);
        println!("Sample: {}", sample(&data, &gpt, device));
    }
    Ok(())
}
