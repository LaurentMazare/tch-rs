use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tch::nn::{FuncT, ModuleT};

#[derive(Debug)]
struct Block {
    block_type: String,
    parameters: BTreeMap<String, String>,
}

impl Block {
    fn get(&self, key: &str) -> failure::Fallible<&str> {
        match self.parameters.get(&key.to_string()) {
            None => bail!("cannot find {} in {}", key, self.block_type),
            Some(value) => Ok(value),
        }
    }
}

#[derive(Debug)]
pub struct Darknet {
    blocks: Vec<Block>,
    parameters: BTreeMap<String, String>,
}

struct Accumulator {
    block_type: Option<String>,
    parameters: BTreeMap<String, String>,
    net: Darknet,
}

impl Accumulator {
    fn new() -> Accumulator {
        Accumulator {
            block_type: None,
            parameters: BTreeMap::new(),
            net: Darknet {
                blocks: vec![],
                parameters: BTreeMap::new(),
            },
        }
    }

    fn finish_block(&mut self) {
        match &self.block_type {
            None => (),
            Some(block_type) => {
                if block_type == "net" {
                    self.net.parameters = self.parameters.clone();
                } else {
                    let block = Block {
                        block_type: block_type.to_string(),
                        parameters: self.parameters.clone(),
                    };
                    self.net.blocks.push(block);
                }
                self.parameters.clear();
            }
        }
        self.block_type = None;
    }
}

pub fn parse_config<T: AsRef<Path>>(path: T) -> failure::Fallible<Darknet> {
    let file = File::open(path.as_ref())?;
    let mut acc = Accumulator::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with("#") {
            continue;
        }
        let line = line.trim();
        if line.starts_with("[") {
            ensure!(line.ends_with("]"), "line does not end with ']' {}", line);
            let line = &line[1..line.len() - 1];
            acc.finish_block();
            acc.block_type = Some(line.to_string());
        } else {
            let key_value: Vec<&str> = line.splitn(2, "=").collect();
            ensure!(key_value.len() == 2, "missing equal {}", line);
            let prev = acc
                .parameters
                .insert(key_value[0].to_owned(), key_value[1].to_owned());
            ensure!(prev == None, "multiple value for key {}", line);
        }
    }
    acc.finish_block();
    Ok(acc.net)
}

enum Bl {
    Layers(Vec<Box<dyn ModuleT>>),
    Route(Vec<usize>),
    Shortcut(usize),
    Yolo((i64, Vec<(i64, i64)>)),
}

fn upsample(prev_channels: i64) -> failure::Fallible<(i64, Bl)> {
    let layer = tch::nn::func_t(|xs, _is_training| {
        let (_n, _c, h, w) = xs.size4().unwrap();
        xs.upsample_nearest2d(&[2 * h, 2 * w])
    });
    Ok((prev_channels, Bl::Layers(vec![Box::new(layer)])))
}

fn int_list_of_string(s: &str) -> failure::Fallible<Vec<i64>> {
    let res: Result<Vec<_>, _> = s.split(",").map(|xs| xs.parse::<i64>()).collect();
    Ok(res?)
}

fn usize_of_index(index: usize, i: i64) -> usize {
    if i >= 0 {
        i as usize
    } else {
        (index as i64 + i) as usize
    }
}

fn route(index: usize, p: &Vec<(i64, Bl)>, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let layers = int_list_of_string(block.get("layers")?)?;
    let layers: Vec<usize> = layers
        .into_iter()
        .map(|l| usize_of_index(index, l))
        .collect();
    let channels = layers.iter().map(|&l| p[l].0).sum();
    Ok((channels, Bl::Route(layers)))
}

fn shortcut(index: usize, p: i64, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let from = block.get("from")?.parse::<i64>()?;
    Ok((p, Bl::Shortcut(usize_of_index(index, from))))
}

fn yolo(index: usize, p: i64, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let classes = block.get("classes")?.parse::<i64>()?;
    let flat = int_list_of_string(block.get("anchors")?)?;
    ensure!(flat.len() % 2 == 0, "even number of anchors");
    let anchors: Vec<_> = (0..(flat.len() / 2))
        .map(|i| (flat[2 * i], flat[2 * i + 1]))
        .collect();
    let mask = int_list_of_string(block.get("mask")?)?;
    let anchors = mask.into_iter().map(|i| anchors[i as usize]).collect();
    Ok((p, Bl::Yolo((classes, anchors))))
}

impl Darknet {
    pub fn build_model(&self, vs: &tch::nn::Path) -> failure::Fallible<FuncT> {
        let mut blocks: Vec<(i64, Bl)> = vec![];
        let mut prev_channels: i64 = 3;
        for (index, block) in self.blocks.iter().enumerate() {
            let channels_and_bl = match block.block_type.as_str() {
                "convolutional" => bail!("todo conv"),
                "upsample" => upsample(prev_channels)?,
                "shortcut" => shortcut(index, prev_channels, &block)?,
                "route" => route(index, &blocks, &block)?,
                "yolo" => yolo(index, prev_channels, &block)?,
                otherwise => bail!("unsupported block type {}", otherwise),
            };
            prev_channels = channels_and_bl.0;
            blocks.push(channels_and_bl);
        }
        let func = tch::nn::func_t(|xs, is_training| xs.shallow_clone());
        Ok(func)
    }
}
