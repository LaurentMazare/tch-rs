test: .FORCE
	cargo test

clean: .FORCE
	cargo clean

gen: .FORCE
	cargo run --bin tch-bindgen --manifest-path tch-bindgen/Cargo.toml --release
	rustfmt src/wrappers/tensor_fallible_generated.rs
	rustfmt src/wrappers/tensor_generated.rs
	rustfmt torch-sys/src/c_generated.rs

.FORCE:
