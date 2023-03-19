# Changelog
This documents the main changes to the `tch` crate.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.11.0
### Added
- Adapt to C++ PyTorch library (`libtorch`) version `v2.0.0`.
### Changed
- Update the `half` dependency to version `2`, [646](https://github.com/LaurentMazare/tch-rs/pull/646).

## v0.10.3 - 2023-02-23
### Added
- Add some helper functions in a utils module to check the available devices and versions.

## v0.10.2 - 2023-02-19
### Added
- Add the `eps` and `amsgrad` options to Adam, [600](https://github.com/LaurentMazare/tch-rs/pull/600).
### Changed
- Fix loading of `VarStore` when using `Mps` devices, [623](https://github.com/LaurentMazare/tch-rs/pull/623).
- Use `ureq` instead of `curl` to reduce compile times, [620](https://github.com/LaurentMazare/tch-rs/pull/620).
- Fix the handling of dicts in TorchScript, [597](https://github.com/LaurentMazare/tch-rs/issues/597).

## Unreleased
### Added
- Expose `optimize_for_inference` for `jit` modules.
- Expose `clone` for `jit` modules.
- Expose `is_training` for `jit` modules.

## v0.10.1 - 2022-12-12
### Changed
- Default `vs.load` to use the Python weight format when the file extension is `.pt` or `.bin`.

## v0.10.0 - 2022-12-12
### Added
- Expose functions for setting manual seeds for CUDA devices, [#500](https://github.com/LaurentMazare/tch-rs/pull/500).
- Expose functions for triggering manual sync, [#500](https://github.com/LaurentMazare/tch-rs/pull/500).
- Add some functions to load Python weight files.

### Changed
- Extending the Kaiming initialization, [#573](https://github.com/LaurentMazare/tch-rs/pull/573).

## v0.9.0 - 2022-11-05
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.13.0`.

## v0.8.0 - 2022-07-04
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.12.0`.

## v0.7.2 - 2022-05-16
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.11.0`.

## v0.6.1 - 2021-10-25
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.10.0`.

## v0.5.0 - 2021-06-25
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.9.0`.

## v0.4.1 - 2021-05-08
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.8.1`.
