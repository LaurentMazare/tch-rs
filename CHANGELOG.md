# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
