use tch::{Device, nn::VarStore};

#[test]
fn var_store_entry() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let t1 = root.entry("key")
        .or_zeros(&[3, 1 , 4]);
    let t2 = root.entry("key")
        .or_zeros(&[1, 5 , 9]);

    assert_eq!(t1.size(), &[3, 1, 4]);
    assert_eq!(t2.size(), &[3, 1, 4]);
}
