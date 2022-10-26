// Stable Diffusion implementation inspired by Jonathan Whitaker
// "Grokking Stable Diffusion" notebook.
// https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
//
// In order to run this, first download the following and extract the file in data/
//
// mkdir -p data && cd data
// wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
// gunzip bpe_simple_vocab_16e6.txt.gz
//
// Download and convert the weights:
// wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
// From python, extract the weights and save them as a .npz file.
//   import numpy as np
//   import torch
//   model = torch.load("./pytorch_model.bin")
//   np.savez("./pytorch_model.npz", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
//
// Then use tensor_tools to convert this to a .ot file that tch can use.
//   cargo run --release --example tensor-tools ls ./data/pytorch_model.npz ./data/pytorch_model.ot
// TODO: fix tensor_tools so that it works properly there.

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use tch::{nn, nn::Module, Device, Kind, Tensor};

// The config details can be found in the "text_config" section of this json file:
// https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
//   "hidden_act": "quick_gelu"
const VOCAB_SIZE: i64 = 49408;
const EMBED_DIM: i64 = 768; // a.k.a. config.hidden_size
const INTERMEDIATE_SIZE: i64 = 3072;
const MAX_POSITION_EMBEDDINGS: i64 = 77;
const NUM_HIDDEN_LAYERS: i64 = 12;
const NUM_ATTENTION_HEADS: i64 = 12;

const BYTES_TO_UNICODE: [(u8, char); 256] = [
    (33, '!'),
    (34, '"'),
    (35, '#'),
    (36, '$'),
    (37, '%'),
    (38, '&'),
    (39, '\''),
    (40, '('),
    (41, ')'),
    (42, '*'),
    (43, '+'),
    (44, ','),
    (45, '-'),
    (46, '.'),
    (47, '/'),
    (48, '0'),
    (49, '1'),
    (50, '2'),
    (51, '3'),
    (52, '4'),
    (53, '5'),
    (54, '6'),
    (55, '7'),
    (56, '8'),
    (57, '9'),
    (58, ':'),
    (59, ';'),
    (60, '<'),
    (61, '='),
    (62, '>'),
    (63, '?'),
    (64, '@'),
    (65, 'A'),
    (66, 'B'),
    (67, 'C'),
    (68, 'D'),
    (69, 'E'),
    (70, 'F'),
    (71, 'G'),
    (72, 'H'),
    (73, 'I'),
    (74, 'J'),
    (75, 'K'),
    (76, 'L'),
    (77, 'M'),
    (78, 'N'),
    (79, 'O'),
    (80, 'P'),
    (81, 'Q'),
    (82, 'R'),
    (83, 'S'),
    (84, 'T'),
    (85, 'U'),
    (86, 'V'),
    (87, 'W'),
    (88, 'X'),
    (89, 'Y'),
    (90, 'Z'),
    (91, '['),
    (92, '\\'),
    (93, ']'),
    (94, '^'),
    (95, '_'),
    (96, '`'),
    (97, 'a'),
    (98, 'b'),
    (99, 'c'),
    (100, 'd'),
    (101, 'e'),
    (102, 'f'),
    (103, 'g'),
    (104, 'h'),
    (105, 'i'),
    (106, 'j'),
    (107, 'k'),
    (108, 'l'),
    (109, 'm'),
    (110, 'n'),
    (111, 'o'),
    (112, 'p'),
    (113, 'q'),
    (114, 'r'),
    (115, 's'),
    (116, 't'),
    (117, 'u'),
    (118, 'v'),
    (119, 'w'),
    (120, 'x'),
    (121, 'y'),
    (122, 'z'),
    (123, '{'),
    (124, '|'),
    (125, '}'),
    (126, '~'),
    (161, '¡'),
    (162, '¢'),
    (163, '£'),
    (164, '¤'),
    (165, '¥'),
    (166, '¦'),
    (167, '§'),
    (168, '¨'),
    (169, '©'),
    (170, 'ª'),
    (171, '«'),
    (172, '¬'),
    (174, '®'),
    (175, '¯'),
    (176, '°'),
    (177, '±'),
    (178, '²'),
    (179, '³'),
    (180, '´'),
    (181, 'µ'),
    (182, '¶'),
    (183, '·'),
    (184, '¸'),
    (185, '¹'),
    (186, 'º'),
    (187, '»'),
    (188, '¼'),
    (189, '½'),
    (190, '¾'),
    (191, '¿'),
    (192, 'À'),
    (193, 'Á'),
    (194, 'Â'),
    (195, 'Ã'),
    (196, 'Ä'),
    (197, 'Å'),
    (198, 'Æ'),
    (199, 'Ç'),
    (200, 'È'),
    (201, 'É'),
    (202, 'Ê'),
    (203, 'Ë'),
    (204, 'Ì'),
    (205, 'Í'),
    (206, 'Î'),
    (207, 'Ï'),
    (208, 'Ð'),
    (209, 'Ñ'),
    (210, 'Ò'),
    (211, 'Ó'),
    (212, 'Ô'),
    (213, 'Õ'),
    (214, 'Ö'),
    (215, '×'),
    (216, 'Ø'),
    (217, 'Ù'),
    (218, 'Ú'),
    (219, 'Û'),
    (220, 'Ü'),
    (221, 'Ý'),
    (222, 'Þ'),
    (223, 'ß'),
    (224, 'à'),
    (225, 'á'),
    (226, 'â'),
    (227, 'ã'),
    (228, 'ä'),
    (229, 'å'),
    (230, 'æ'),
    (231, 'ç'),
    (232, 'è'),
    (233, 'é'),
    (234, 'ê'),
    (235, 'ë'),
    (236, 'ì'),
    (237, 'í'),
    (238, 'î'),
    (239, 'ï'),
    (240, 'ð'),
    (241, 'ñ'),
    (242, 'ò'),
    (243, 'ó'),
    (244, 'ô'),
    (245, 'õ'),
    (246, 'ö'),
    (247, '÷'),
    (248, 'ø'),
    (249, 'ù'),
    (250, 'ú'),
    (251, 'û'),
    (252, 'ü'),
    (253, 'ý'),
    (254, 'þ'),
    (255, 'ÿ'),
    (0, 'Ā'),
    (1, 'ā'),
    (2, 'Ă'),
    (3, 'ă'),
    (4, 'Ą'),
    (5, 'ą'),
    (6, 'Ć'),
    (7, 'ć'),
    (8, 'Ĉ'),
    (9, 'ĉ'),
    (10, 'Ċ'),
    (11, 'ċ'),
    (12, 'Č'),
    (13, 'č'),
    (14, 'Ď'),
    (15, 'ď'),
    (16, 'Đ'),
    (17, 'đ'),
    (18, 'Ē'),
    (19, 'ē'),
    (20, 'Ĕ'),
    (21, 'ĕ'),
    (22, 'Ė'),
    (23, 'ė'),
    (24, 'Ę'),
    (25, 'ę'),
    (26, 'Ě'),
    (27, 'ě'),
    (28, 'Ĝ'),
    (29, 'ĝ'),
    (30, 'Ğ'),
    (31, 'ğ'),
    (32, 'Ġ'),
    (127, 'ġ'),
    (128, 'Ģ'),
    (129, 'ģ'),
    (130, 'Ĥ'),
    (131, 'ĥ'),
    (132, 'Ħ'),
    (133, 'ħ'),
    (134, 'Ĩ'),
    (135, 'ĩ'),
    (136, 'Ī'),
    (137, 'ī'),
    (138, 'Ĭ'),
    (139, 'ĭ'),
    (140, 'Į'),
    (141, 'į'),
    (142, 'İ'),
    (143, 'ı'),
    (144, 'Ĳ'),
    (145, 'ĳ'),
    (146, 'Ĵ'),
    (147, 'ĵ'),
    (148, 'Ķ'),
    (149, 'ķ'),
    (150, 'ĸ'),
    (151, 'Ĺ'),
    (152, 'ĺ'),
    (153, 'Ļ'),
    (154, 'ļ'),
    (155, 'Ľ'),
    (156, 'ľ'),
    (157, 'Ŀ'),
    (158, 'ŀ'),
    (159, 'Ł'),
    (160, 'ł'),
    (173, 'Ń'),
];

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

// This is mostly a Rust rewrite of the original Python CLIP code.
// https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
struct Tokenizer {
    re: regex::Regex,
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    start_of_text_token: usize,
    end_of_text_token: usize,
}

impl Tokenizer {
    fn create<T: AsRef<std::path::Path>>(bpe_path: T) -> anyhow::Result<Tokenizer> {
        let bpe_file = std::fs::File::open(bpe_path)?;
        let bpe_lines: Result<Vec<String>, _> = std::io::BufReader::new(bpe_file).lines().collect();
        let bpe_lines = bpe_lines?;
        let bpe_lines: Result<Vec<_>, _> = bpe_lines[1..49152 - 256 - 2 + 1]
            .into_iter()
            .map(|line| {
                let vs: Vec<_> = line.split_whitespace().collect();
                if vs.len() != 2 {
                    anyhow::bail!("expected two items got {} '{}'", vs.len(), line)
                }
                Ok((vs[0].to_string(), vs[1].to_string()))
            })
            .collect();
        let bpe_lines = bpe_lines?;
        let mut vocab: Vec<String> = Vec::new();
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(elem.into())
        }
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(format!("{}</w>", elem));
        }
        for elem in bpe_lines.iter() {
            vocab.push(format!("{}{}", elem.0, elem.1))
        }
        let start_of_text_token = vocab.len();
        vocab.push("<|startoftext|>".to_string());
        let end_of_text_token = vocab.len();
        vocab.push("<|endoftext|>".to_string());
        let encoder: HashMap<_, _> = vocab.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let decoder: HashMap<_, _> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        let bpe_ranks: HashMap<_, _> =
            bpe_lines.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let re = regex::Regex::new(PAT)?;
        let tokenizer =
            Tokenizer { encoder, re, bpe_ranks, decoder, start_of_text_token, end_of_text_token };
        Ok(tokenizer)
    }

    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        for (i, v) in word.iter().enumerate() {
            if i > 0 {
                pairs.insert((word[i - 1].clone(), v.clone()));
            }
        }
        pairs
    }

    fn bpe(&self, token: &str) -> Vec<usize> {
        let mut word: Vec<String> = token.chars().map(|x| x.to_string()).collect();
        if word.is_empty() {
            return Vec::new();
        }
        let last_index = word.len() - 1;
        word[last_index] = format!("{}</w>", word[last_index]);
        while word.len() > 1 {
            let mut current_min = None;
            let pairs = Self::get_pairs(&word);
            for p in pairs.iter() {
                match self.bpe_ranks.get(&p) {
                    None => {}
                    Some(v) => {
                        let should_replace = match current_min {
                            None => true,
                            Some((current_min, _)) => v < current_min,
                        };
                        if should_replace {
                            current_min = Some((v, p))
                        }
                    }
                }
            }
            let (first, second) = match current_min {
                None => break,
                Some((_v, (first, second))) => (first, second),
            };
            let mut new_word = vec![];
            let mut index = 0;
            while index < word.len() {
                let w = &word[index];
                if index + 1 < word.len() && w == first && &word[index + 1] == second {
                    new_word.push(format!("{}{}", first, second));
                    index += 2
                } else {
                    new_word.push(w.clone());
                    index += 1
                }
            }
            word = new_word
        }
        word.iter().map(|x| *self.encoder.get(x).unwrap()).collect()
    }

    fn encode(&self, s: &str, pad_size_to: Option<usize>) -> anyhow::Result<Vec<usize>> {
        let s = s.to_lowercase();
        let mut bpe_tokens: Vec<usize> = vec![self.start_of_text_token];
        for token in self.re.captures_iter(&s) {
            let token = token.get(0).unwrap().as_str();
            println!(">>> {:?}", token);
            bpe_tokens.extend(self.bpe(token))
        }
        match pad_size_to {
            None => bpe_tokens.push(self.end_of_text_token),
            Some(pad_size_to) => {
                bpe_tokens.resize_with(
                    std::cmp::min(bpe_tokens.len(), pad_size_to - 1),
                    Default::default,
                );
                while bpe_tokens.len() < pad_size_to {
                    bpe_tokens.push(self.end_of_text_token)
                }
            }
        }
        Ok(bpe_tokens)
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let s: String = tokens.iter().map(|token| self.decoder[token].as_str()).collect();
        s.replace("</w>", " ")
    }
}

// CLIP Text Model
// https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py
#[derive(Debug)]
struct ClipTextEmbeddings {
    token_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    position_ids: Tensor,
}

impl ClipTextEmbeddings {
    fn new(vs: nn::Path) -> Self {
        let token_embedding =
            nn::embedding(&vs / "token_embedding", VOCAB_SIZE, EMBED_DIM, Default::default());
        let position_embedding = nn::embedding(
            &vs / "position_embedding",
            MAX_POSITION_EMBEDDINGS,
            EMBED_DIM,
            Default::default(),
        );
        let position_ids = Tensor::arange(MAX_POSITION_EMBEDDINGS, (Kind::Int64, vs.device()))
            .expand(&[1, -1], false);
        ClipTextEmbeddings { token_embedding, position_embedding, position_ids }
    }
}

impl Module for ClipTextEmbeddings {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let token_embedding = self.token_embedding.forward(&xs);
        let position_embedding = self.position_embedding.forward(&self.position_ids);
        token_embedding + position_embedding
    }
}

fn quick_gelu(xs: &Tensor) -> Tensor {
    xs * (xs * 1.702).sigmoid()
}

#[derive(Debug)]
struct ClipAttention {
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    q_proj: nn::Linear,
    out_proj: nn::Linear,
    head_dim: i64,
    scale: f64,
}

impl ClipAttention {
    fn new(vs: nn::Path) -> Self {
        let k_proj = nn::linear(&vs / "k_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let v_proj = nn::linear(&vs / "v_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let q_proj = nn::linear(&vs / "q_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let out_proj = nn::linear(&vs / "out_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let head_dim = EMBED_DIM / NUM_ATTENTION_HEADS;
        let scale = (head_dim as f64).powf(-0.5);
        ClipAttention { k_proj, v_proj, q_proj, out_proj, head_dim, scale }
    }

    fn shape(&self, xs: &Tensor, seq_len: i64, bsz: i64) -> Tensor {
        xs.view((bsz, seq_len, NUM_ATTENTION_HEADS, self.head_dim)).transpose(1, 2).contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let (bsz, tgt_len, embed_dim) = xs.size3().unwrap();
        let query_states = xs.apply(&self.q_proj) * self.scale;
        let proj_shape = (bsz * NUM_ATTENTION_HEADS, -1, self.head_dim);
        let query_states = self.shape(&query_states, tgt_len, bsz).view(proj_shape.clone());
        let key_states = self.shape(&xs.apply(&self.k_proj), -1, bsz).view(proj_shape.clone());
        let value_states = self.shape(&xs.apply(&self.v_proj), -1, bsz).view(proj_shape.clone());
        let attn_weights = query_states.bmm(&key_states.transpose(1, 2));

        let src_len = key_states.size()[1];
        let attn_weights =
            attn_weights.view((bsz, NUM_ATTENTION_HEADS, tgt_len, src_len)) + causal_attention_mask;
        let attn_weights = attn_weights.view((bsz * NUM_ATTENTION_HEADS, tgt_len, src_len));
        let attn_weights = attn_weights.softmax(-1, Kind::Float);

        let attn_output = attn_weights.bmm(&value_states);
        attn_output
            .view((bsz, NUM_ATTENTION_HEADS, tgt_len, self.head_dim))
            .transpose(1, 2)
            .reshape(&[bsz, tgt_len, embed_dim])
            .apply(&self.out_proj)
    }
}

#[derive(Debug)]
struct ClipMlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ClipMlp {
    fn new(vs: nn::Path) -> Self {
        let fc1 = nn::linear(&vs / "fc1", EMBED_DIM, INTERMEDIATE_SIZE, Default::default());
        let fc2 = nn::linear(&vs / "fc2", INTERMEDIATE_SIZE, EMBED_DIM, Default::default());
        ClipMlp { fc1, fc2 }
    }
}

impl Module for ClipMlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.apply(&self.fc1);
        quick_gelu(&xs).apply(&self.fc2)
    }
}

#[derive(Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: nn::Path) -> Self {
        let self_attn = ClipAttention::new(&vs / "self_attn");
        let layer_norm1 = nn::layer_norm(&vs / "layer_norm1", vec![EMBED_DIM], Default::default());
        let mlp = ClipMlp::new(&vs / "mlp");
        let layer_norm2 = nn::layer_norm(&vs / "layer_norm2", vec![EMBED_DIM], Default::default());
        ClipEncoderLayer { self_attn, layer_norm1, mlp, layer_norm2 }
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs);
        let xs = self.self_attn.forward(&xs, &causal_attention_mask);
        let xs = xs + residual;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs);
        let xs = self.mlp.forward(&xs);
        xs + residual
    }
}

#[derive(Debug)]
struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    fn new(vs: nn::Path) -> Self {
        let vs = &vs / "layers";
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..NUM_HIDDEN_LAYERS {
            let layer = ClipEncoderLayer::new(&vs / index);
            layers.push(layer)
        }
        ClipEncoder { layers }
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)
        }
        xs
    }
}

#[derive(Debug)]
struct ClipTextTransformer {
    embeddings: ClipTextEmbeddings,
    encoder: ClipEncoder,
    final_layer_norm: nn::LayerNorm,
}

impl ClipTextTransformer {
    fn new(vs: nn::Path) -> Self {
        let vs = &vs / "text_model";
        let embeddings = ClipTextEmbeddings::new(&vs / "embeddings");
        let encoder = ClipEncoder::new(&vs / "encoder");
        let final_layer_norm =
            nn::layer_norm(&vs / "final_layer_norm", vec![EMBED_DIM], Default::default());
        ClipTextTransformer { embeddings, encoder, final_layer_norm }
    }

    // https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
    fn build_causal_attention_mask(bsz: i64, seq_len: i64, device: Device) -> Tensor {
        let mut mask = Tensor::ones(&[bsz, seq_len, seq_len], (Kind::Float, device));
        mask.fill_(f32::MIN as f64).triu_(1).unsqueeze(1)
    }
}

impl Module for ClipTextTransformer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (bsz, seq_len) = xs.size2().unwrap();
        let xs = self.embeddings.forward(xs);
        let causal_attention_mask = Self::build_causal_attention_mask(bsz, seq_len, xs.device());
        let xs = self.encoder.forward(&xs, &causal_attention_mask);
        xs.apply(&self.final_layer_norm)
    }
}

#[derive(Debug, Clone, Copy)]
struct AttentionBlockConfig {
    num_head_channels: Option<i64>,
    num_groups: i64,
    rescale_output_factor: f64,
    eps: f64,
}

impl Default for AttentionBlockConfig {
    fn default() -> Self {
        Self { num_head_channels: None, num_groups: 32, rescale_output_factor: 1., eps: 1e-5 }
    }
}

#[derive(Debug)]
struct AttentionBlock {
    group_norm: nn::GroupNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    proj_attn: nn::Linear,
    channels: i64,
    num_heads: i64,
    config: AttentionBlockConfig,
}

impl AttentionBlock {
    fn new(vs: nn::Path, channels: i64, config: AttentionBlockConfig) -> Self {
        let num_head_channels = config.num_head_channels.unwrap_or(channels);
        let num_heads = channels / num_head_channels;
        let group_cfg = nn::GroupNormConfig { eps: config.eps, affine: true, ..Default::default() };
        let group_norm = nn::group_norm(&vs / "group_norm", config.num_groups, channels, group_cfg);
        let query = nn::linear(&vs / "linear", channels, channels, Default::default());
        let key = nn::linear(&vs / "key", channels, channels, Default::default());
        let value = nn::linear(&vs / "value", channels, channels, Default::default());
        let proj_attn = nn::linear(&vs / "proj_attn", channels, channels, Default::default());
        Self { group_norm, query, key, value, proj_attn, channels, num_heads, config }
    }

    fn transpose_for_scores(&self, xs: Tensor) -> Tensor {
        let (batch, t, _h_times_d) = xs.size3().unwrap();
        xs.view((batch, t, self.num_heads, -1)).permute(&[0, 2, 1, 3])
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let residual = xs;
        let (batch, channel, height, width) = xs.size4().unwrap();
        let xs = xs.apply(&self.group_norm).view((batch, channel, height * width)).transpose(1, 2);

        let query_proj = xs.apply(&self.query);
        let key_proj = xs.apply(&self.key);
        let value_proj = xs.apply(&self.value);

        let query_states = self.transpose_for_scores(query_proj);
        let key_states = self.transpose_for_scores(key_proj);
        let value_states = self.transpose_for_scores(value_proj);

        let scale = f64::powf((self.channels as f64) / (self.num_heads as f64), -0.25);
        let attention_scores =
            (query_states * scale).matmul(&(key_states.transpose(-1, -2) * scale));
        let attention_probs = attention_scores.softmax(-1, Kind::Float);

        let xs = attention_probs.matmul(&value_states);
        let xs = xs.permute(&[0, 2, 1, 3]).contiguous();
        let mut new_xs_shape = xs.size();
        new_xs_shape.pop();
        new_xs_shape.pop();
        new_xs_shape.push(self.channels);

        xs.view(new_xs_shape.as_slice())
            .apply(&self.proj_attn)
            .transpose(-1, -2)
            .view((batch, channel, height, width))
            / self.config.rescale_output_factor
    }
}

#[derive(Debug)]
struct Downsample2D {
    conv: Option<nn::Conv2D>,
    padding: i64,
}

impl Downsample2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        use_conv: bool,
        out_channels: i64,
        padding: i64,
    ) -> Self {
        let conv = if use_conv {
            let config = nn::ConvConfig { stride: 2, padding, ..Default::default() };
            let conv = nn::conv2d(&vs / "conv", in_channels, out_channels, 3, config);
            Some(conv)
        } else {
            None
        };
        Downsample2D { conv, padding }
    }
}

impl Module for Downsample2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match &self.conv {
            None => xs.avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, true, None),
            Some(conv) => {
                if self.padding == 0 {
                    xs.pad(&[0, 1, 0, 1], "constant", Some(0.)).apply(conv)
                } else {
                    xs.apply(conv)
                }
            }
        }
    }
}

// This does not support the conv-transpose mode.
#[derive(Debug)]
struct Upsample2D {
    conv: nn::Conv2D,
}

impl Upsample2D {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let config = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv = nn::conv2d(&vs / "conv", in_channels, out_channels, 3, config);
        Self { conv }
    }
}

impl Module for Upsample2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.upsample_nearest2d(&[], Some(2.), Some(2.)).apply(&self.conv)
    }
}

#[derive(Debug, Clone, Copy)]
struct ResnetBlock2DConfig {
    out_channels: Option<i64>,
    conv_shortcut: bool,
    // temb_channels: None
    groups: i64,
    groups_out: Option<i64>,
    pre_norm: bool,
    eps: f64,
    use_in_shortcut: Option<bool>,
    // non_linearity: silu
    output_scale_factor: f64,
}

impl Default for ResnetBlock2DConfig {
    fn default() -> Self {
        Self {
            out_channels: None,
            conv_shortcut: false,
            groups: 32,
            groups_out: None,
            pre_norm: true,
            eps: 1e-6,
            use_in_shortcut: None,
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
struct ResnetBlock2D {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2D,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2D,
    conv_shortcut: Option<nn::Conv2D>,
    use_in_shortcut: bool,
    config: ResnetBlock2DConfig,
}

impl ResnetBlock2D {
    fn new(vs: nn::Path, in_channels: i64, config: ResnetBlock2DConfig) -> Self {
        let out_channels = config.out_channels.unwrap_or(in_channels);
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let group_cfg = nn::GroupNormConfig { eps: config.eps, affine: true, ..Default::default() };
        let norm1 = nn::group_norm(&vs / "norm1", config.groups, in_channels, group_cfg);
        let conv1 = nn::conv2d(&vs / "conv1", in_channels, out_channels, 3, conv_cfg);
        let groups_out = config.groups_out.unwrap_or(config.groups);
        let norm2 = nn::group_norm(&vs / "norm2", groups_out, out_channels, group_cfg);
        let conv2 = nn::conv2d(&vs / "conv2", out_channels, out_channels, 3, conv_cfg);
        let use_in_shortcut = config.use_in_shortcut.unwrap_or(in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = nn::ConvConfig { stride: 1, padding: 0, ..Default::default() };
            Some(nn::conv2d(&vs / "conv_shortcut", in_channels, out_channels, 1, conv_cfg))
        } else {
            None
        };
        Self { norm1, conv1, norm2, conv2, config, conv_shortcut, use_in_shortcut }
    }
}

impl Module for ResnetBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let shortcut_xs = match &self.conv_shortcut {
            Some(conv_shortcut) => xs.apply(conv_shortcut),
            None => xs.shallow_clone(),
        };
        let xs = xs
            .apply(&self.norm1)
            .silu()
            .apply(&self.conv1)
            .apply(&self.norm2)
            .silu()
            .apply(&self.conv2);
        (shortcut_xs + xs) / self.config.output_scale_factor
    }
}

#[derive(Debug, Clone, Copy)]
struct DownEncoderBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    resnet_groups: i64,
    resnet_pre_norm: bool,
    output_scale_factor: f64,
    add_downsample: bool,
    downsample_padding: i64,
}

impl Default for DownEncoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            resnet_pre_norm: true,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    config: DownEncoderBlock2DConfig,
}

impl DownEncoderBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: DownEncoderBlock2DConfig,
    ) -> Self {
        let resnets: Vec<_> = {
            let vs = &vs / "resnets";
            let conv_cfg = ResnetBlock2DConfig {
                eps: config.resnet_eps,
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                pre_norm: config.resnet_pre_norm,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(&vs / i, in_channels, conv_cfg)
                })
                .collect()
        };
        let downsampler = if config.add_downsample {
            let downsample = Downsample2D::new(
                &vs / "downsample",
                in_channels,
                true,
                out_channels,
                config.downsample_padding,
            );
            Some(downsample)
        } else {
            None
        };
        Self { resnets, downsampler, config }
    }
}

impl Module for DownEncoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for resnet in self.resnets.iter() {
            xs = xs.apply(resnet)
        }
        match &self.downsampler {
            Some(downsampler) => xs.apply(downsampler),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UpDecoderBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    resnet_groups: i64,
    resnet_pre_norm: bool,
    output_scale_factor: f64,
    add_upsample: bool,
}

impl Default for UpDecoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            resnet_pre_norm: true,
            output_scale_factor: 1.,
            add_upsample: true,
        }
    }
}

#[derive(Debug)]
struct UpDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
    config: UpDecoderBlock2DConfig,
}

impl UpDecoderBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: UpDecoderBlock2DConfig,
    ) -> Self {
        let resnets: Vec<_> = {
            let vs = &vs / "resnets";
            let conv_cfg = ResnetBlock2DConfig {
                eps: config.resnet_eps,
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                pre_norm: config.resnet_pre_norm,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(&vs / i, in_channels, conv_cfg)
                })
                .collect()
        };
        let upsampler = if config.add_upsample {
            let upsample = Upsample2D::new(&vs / "upsample", out_channels, out_channels);
            Some(upsample)
        } else {
            None
        };
        Self { resnets, upsampler, config }
    }
}

impl Module for UpDecoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for resnet in self.resnets.iter() {
            xs = xs.apply(resnet)
        }
        match &self.upsampler {
            Some(upsampler) => xs.apply(upsampler),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UNetMidBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    resnet_groups: Option<i64>,
    resnet_pre_norm: bool,
    attn_num_head_channels: Option<i64>,
    // attention_type "default"
    output_scale_factor: f64,
}

impl Default for UNetMidBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            resnet_pre_norm: true,
            attn_num_head_channels: Some(1),
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
struct UNetMidBlock2D {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(AttentionBlock, ResnetBlock2D)>,
    config: UNetMidBlock2DConfig,
}

impl UNetMidBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        // temb_channels: None
        config: UNetMidBlock2DConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let vs_attns = &vs / "attentions";
        let resnet_groups = config.resnet_groups.unwrap_or(i64::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            pre_norm: config.resnet_pre_norm,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(&vs_resnets / "0", in_channels, resnet_cfg);
        let attn_cfg = AttentionBlockConfig {
            num_head_channels: config.attn_num_head_channels,
            num_groups: resnet_groups,
            rescale_output_factor: config.output_scale_factor,
            eps: config.resnet_eps,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = AttentionBlock::new(&vs_attns / index, in_channels, attn_cfg);
            let resnet = ResnetBlock2D::new(&vs_resnets / (index + 1), in_channels, resnet_cfg);
            attn_resnets.push((attn, resnet))
        }
        Self { resnet, attn_resnets, config }
    }
}

impl Module for UNetMidBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.resnet);
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = xs.apply(attn).apply(resnet)
        }
        xs
    }
}

#[derive(Debug, Clone)]
struct EncoderConfig {
    // down_block_types: DownEncoderBlock2D
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    norm_num_groups: i64,
    double_z: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
            double_z: true,
        }
    }
}

#[derive(Debug)]
struct Encoder {
    conv_in: nn::Conv2D,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    config: EncoderConfig,
}

impl Encoder {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: EncoderConfig) -> Self {
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in =
            nn::conv2d(&vs / "conv_in", in_channels, config.block_out_channels[0], 3, conv_cfg);
        let mut down_blocks = vec![];
        let vs_down_blocks = &vs / "down_blocks";
        for index in 0..config.block_out_channels.len() {
            let out_channels = config.block_out_channels[index];
            let in_channels = if index > 0 {
                config.block_out_channels[index - 1]
            } else {
                config.block_out_channels[0]
            };
            let is_final = index + 1 == config.block_out_channels.len();
            let cfg = DownEncoderBlock2DConfig {
                num_layers: config.layers_per_block,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_downsample: !is_final,
                downsample_padding: 0,
                ..Default::default()
            };
            let down_block =
                DownEncoderBlock2D::new(&vs_down_blocks / index, in_channels, out_channels, cfg);
            down_blocks.push(down_block)
        }
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block = UNetMidBlock2D::new(&vs / "mid_block", last_block_out_channels, mid_cfg);
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, ..Default::default() };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            last_block_out_channels,
            group_cfg,
        );
        let conv_out_channels = if config.double_z { 2 * out_channels } else { out_channels };
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_out =
            nn::conv2d(&vs / "conv_out", last_block_out_channels, conv_out_channels, 3, conv_cfg);
        Self { conv_in, down_blocks, mid_block, conv_norm_out, conv_out, config }
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.conv_in);
        for down_block in self.down_blocks.iter() {
            xs = xs.apply(down_block)
        }
        xs.apply(&self.mid_block).apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct DecoderConfig {
    // up_block_types: UpDecoderBlock2D
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    norm_num_groups: i64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self { block_out_channels: vec![64], layers_per_block: 2, norm_num_groups: 32 }
    }
}

#[derive(Debug)]
struct Decoder {
    conv_in: nn::Conv2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    config: DecoderConfig,
}

impl Decoder {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: DecoderConfig) -> Self {
        let n_block_out_channels = config.block_out_channels.len();
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in =
            nn::conv2d(&vs / "conv_in", in_channels, last_block_out_channels, 3, conv_cfg);
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block = UNetMidBlock2D::new(&vs / "mid_block", last_block_out_channels, mid_cfg);
        let mut up_blocks = vec![];
        let vs_up_blocks = &vs / "up_blocks";
        let reversed_block_out_channels: Vec<_> =
            config.block_out_channels.iter().map(|x| *x).rev().collect();
        for index in 0..n_block_out_channels {
            let out_channels = reversed_block_out_channels[index];
            let in_channels = if index > 0 {
                reversed_block_out_channels[index - 1]
            } else {
                reversed_block_out_channels[0]
            };
            let is_final = index + 1 == n_block_out_channels;
            let cfg = UpDecoderBlock2DConfig {
                num_layers: config.layers_per_block + 1,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_upsample: !is_final,
                ..Default::default()
            };
            let up_block =
                UpDecoderBlock2D::new(&vs_up_blocks / index, in_channels, out_channels, cfg);
            up_blocks.push(up_block)
        }
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, ..Default::default() };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            config.block_out_channels[0],
            group_cfg,
        );
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_out =
            nn::conv2d(&vs / "conv_out", config.block_out_channels[0], out_channels, 3, conv_cfg);
        Self { conv_in, up_blocks, mid_block, conv_norm_out, conv_out, config }
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.conv_in).apply(&self.mid_block);
        for up_block in self.up_blocks.iter() {
            xs = xs.apply(up_block)
        }
        xs.apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct AutoEncoderKLConfig {
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    latent_channels: i64,
    norm_num_groups: i64,
}

impl Default for AutoEncoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 32,
        }
    }
}

// https://github.com/huggingface/diffusers/blob/970e30606c2944e3286f56e8eb6d3dc6d1eb85f7/src/diffusers/models/vae.py#L485
// This implementation is specific to the config used in stable-diffusion-v1-4
// https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
#[derive(Debug)]
struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: nn::Conv2D,
    post_quant_conv: nn::Conv2D,
    config: AutoEncoderKLConfig,
}

impl AutoEncoderKL {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: AutoEncoderKLConfig) -> Self {
        let latent_channels = config.latent_channels;
        let encoder_cfg = EncoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
            double_z: true,
        };
        let encoder = Encoder::new(&vs / "encoder", in_channels, latent_channels, encoder_cfg);
        let decoder_cfg = DecoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
        };
        let decoder = Decoder::new(&vs / "decoder", latent_channels, out_channels, decoder_cfg);
        let conv_cfg = Default::default();
        let quant_conv =
            nn::conv2d(&vs / "quant_conv", 2 * latent_channels, 2 * latent_channels, 1, conv_cfg);
        let post_quant_conv =
            nn::conv2d(&vs / "post_quant_conv", latent_channels, latent_channels, 1, conv_cfg);
        Self { encoder, decoder, quant_conv, post_quant_conv, config }
    }

    // Returns the parameters of the latent distribution.
    fn encode(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.encoder).apply(&self.quant_conv)
    }

    /// Takes as input some sampled values.
    fn decode(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.post_quant_conv).apply(&self.decoder)
    }
}

// TODO: UNet2DConditionModel
// https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
// https://github.com/huggingface/diffusers/blob/970e30606c2944e3286f56e8eb6d3dc6d1eb85f7/src/diffusers/models/unet_2d_condition.py#L37
// TODO: LMSDiscreteScheduler
// https://github.com/huggingface/diffusers/blob/32bf4fdc4386809c870528cb261028baae012d27/src/diffusers/schedulers/scheduling_lms_discrete.py#L47

fn main() -> anyhow::Result<()> {
    tch::maybe_init_cuda();
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    let tokenizer = Tokenizer::create("data/bpe_simple_vocab_16e6.txt")?;
    let tokens = tokenizer.encode("This is some sample text to be tokenized.", Some(77))?;
    println!("Tokens: {:?}", tokens);
    let str = tokenizer.decode(&tokens);
    println!("Str: {}", str);
    let tokens = tokenizer.encode("Cave painting of a bird, flooble", Some(77))?;
    println!("Tokens: {:?}", tokens);
    let str = tokenizer.decode(&tokens);
    println!("Str: {}", str);

    let _no_grad_guard = tch::no_grad_guard();
    let mut vs = nn::VarStore::new(Device::Cpu);
    let text_model = ClipTextTransformer::new(vs.root());
    vs.load("data/pytorch_model.ot")?;
    let tokens: Vec<i64> = tokens.iter().map(|x| *x as i64).collect();
    let tokens = Tensor::of_slice(&tokens).view((1, -1));
    println!("Input tensor {:?}", tokens.size());
    let text_embeddings = text_model.forward(&tokens);
    println!("Embedding shape {:?}", text_embeddings.size());

    let mut vs = nn::VarStore::new(Device::Cpu);
    // https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
    let autoencoder_cfg = AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
    };
    let _autoencoder = AutoEncoderKL::new(vs.root(), 3, 3, autoencoder_cfg);
    Ok(())
}
