use tch::{MaskSelectOp, Tensor};

#[test]
fn tensor_masking() {
    // array
    {
        let input: Tensor = [[1i64, 2], [3, 4], [5, 6]].try_into().unwrap();
        let mask = [[true, false], [false, false], [false, true]];
        let output: Tensor = input.m(mask);
        let expect: Tensor = [1i64, 6].try_into().unwrap();
        assert_eq!(output, expect);
    }

    // slice
    {
        let input: Tensor = [[1i64, 2], [3, 4], [5, 6]].try_into().unwrap();
        let mask = [false, true].as_slice();
        let output: Tensor = input.m(mask);
        let expect: Tensor = [2i64, 4, 6].try_into().unwrap();
        assert_eq!(output, expect);
    }

    // Tensor
    {
        let input: Tensor = [[1i64, 2], [3, 4], [5, 6]].try_into().unwrap();
        let mask: Tensor = [[true, false], [false, false], [false, true]].try_into().unwrap();
        let output: Tensor = input.m(mask);
        let expect: Tensor = [1i64, 6].try_into().unwrap();
        assert_eq!(output, expect);
    }

    // &Tensor
    {
        let input: Tensor = [[1i64, 2], [3, 4], [5, 6]].try_into().unwrap();
        let mask: Tensor = [[true, false], [false, false], [false, true]].try_into().unwrap();
        let output: Tensor = input.m(&mask);
        let expect: Tensor = [1i64, 6].try_into().unwrap();
        assert_eq!(output, expect);
    }
}
