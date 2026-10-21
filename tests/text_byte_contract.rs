use tch::{data::TextData, nn, Device, Kind};

#[test]
fn byte_dictionary_seed_and_checkpoint_contract() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::temp_dir().join(format!("tch-byte-contract-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    let corpus = root.join("bytes.bin");
    let mut bytes = vec![0x41, 0xff, 0x80, 0x41];
    bytes.extend(0..=255u8);
    std::fs::write(&corpus, &bytes)?;
    let data = TextData::new(&corpus)?;
    assert_eq!(data.labels(), 256);
    assert_eq!(data.data().kind(), Kind::Uint8);
    assert_eq!(data.data().size(), [260]);
    assert_eq!(data.char_to_label('A')?, 0);
    assert_eq!(data.char_to_label('\u{ff}')?, 1);
    assert_eq!(data.char_to_label('\u{80}')?, 2);
    for byte in 0..=255u8 {
        let label = data.char_to_label(char::from(byte))?;
        assert_eq!(data.label_to_char(i64::from(label)), char::from(byte));
    }
    let expected: Vec<u8> =
        bytes.iter().map(|&b| data.char_to_label(char::from(b)).unwrap()).collect();
    assert_eq!(Vec::<u8>::try_from(data.data())?, expected);
    let vs = nn::VarStore::new(Device::Cpu);
    let weights = vs.root().var("token_embedding", &[data.labels(), 2], nn::Init::Const(0.75));
    let checkpoint = root.join("model.ot");
    vs.save(&checkpoint)?;
    let restored = TextData::new(&corpus)?;
    let mut restored_vs = nn::VarStore::new(Device::Cpu);
    let restored_weights =
        restored_vs.root().var("token_embedding", &[restored.labels(), 2], nn::Init::Const(0.));
    restored_vs.load(&checkpoint)?;
    assert!(weights.equal(&restored_weights));
    assert!(data.data().equal(&restored.data()));
    let seed: Result<Vec<u8>, _> = "Aÿ".chars().map(|c| restored.char_to_label(c)).collect();
    assert_eq!(seed?, [0, 1]);
    std::fs::remove_file(checkpoint)?;
    std::fs::remove_file(corpus)?;
    std::fs::remove_dir(root)?;
    Ok(())
}

#[test]
fn min_gpt_seed_rejects_non_byte_scalar_instead_of_aliasing(
) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("tch-seed-alias-{}.bin", std::process::id()));
    std::fs::write(&path, b"ABBA")?;
    let data = TextData::new(&path)?;
    std::fs::remove_file(path)?;
    let seed: Result<Vec<u8>, _> = "Ł".chars().map(|c| data.char_to_label(c)).collect();
    assert!(seed.is_err(), "U+0141 must not alias byte A in the min-gpt chars-to-label seed path");
    assert!(data.char_to_label('Z').is_err());
    Ok(())
}
