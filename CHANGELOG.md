# Changelog

## [0.6.0](https://github.com/laurigates/loractl/compare/v0.5.1...v0.6.0) (2026-07-15)


### Features

* **core:** DiffusionTrainer — the end-to-end Krea 2 LoRA trainer (M14) ([#73](https://github.com/laurigates/loractl/issues/73)) ([9188909](https://github.com/laurigates/loractl/commit/9188909b8c12c3a59c80ab322ca0adf08a3fd68b))
* **core:** image dataset pipeline — aspect buckets + latent/conditioning cache (M12) ([#71](https://github.com/laurigates/loractl/issues/71)) ([88972d9](https://github.com/laurigates/loractl/commit/88972d9519d10b4bb43d1b8b397cf0bfd0ebac3d))
* **core:** Krea 2 MMDiT denoiser with official-implementation parity + LoRA attach (M11) ([#70](https://github.com/laurigates/loractl/issues/70)) ([b17d5eb](https://github.com/laurigates/loractl/commit/b17d5eb73b348ee8d2468ab4f4331729989164ea))
* **core:** M13 memory knobs — f16 precision (wgpu) + gradient checkpointing ([#72](https://github.com/laurigates/loractl/issues/72)) ([5c86fcc](https://github.com/laurigates/loractl/commit/5c86fcc5c97996db132660b2eb24c53d493b6a53))
* **core:** Qwen3-VL text-conditioning encoder with transformers parity (M10) ([#69](https://github.com/laurigates/loractl/issues/69)) ([0a36056](https://github.com/laurigates/loractl/commit/0a360562acec9a55cb9b2db7e7983591037425c6))


### Bug Fixes

* **core:** real-run fixes — f16 numerics, adapter resume, candle arm, torch reference trainer (M14) ([#75](https://github.com/laurigates/loractl/issues/75)) ([6078bfd](https://github.com/laurigates/loractl/commit/6078bfdec373a33621f54c071eddbff7387d9dc0))

## [0.5.1](https://github.com/laurigates/loractl/compare/v0.5.0...v0.5.1) (2026-07-12)


### Bug Fixes

* **api:** bound run-registry memory and confine unauthenticated output paths ([#63](https://github.com/laurigates/loractl/issues/63)) ([5369e10](https://github.com/laurigates/loractl/commit/5369e10e4ffdcbfe4e1be0d2b1d223ae23dc49f3))

## [0.5.0](https://github.com/laurigates/loractl/compare/v0.4.1...v0.5.0) (2026-07-12)


### Features

* **just:** add clean recipe for build artifacts ([#57](https://github.com/laurigates/loractl/issues/57)) ([0f2489c](https://github.com/laurigates/loractl/commit/0f2489c51b89dd048bacc54571e4de413c04a8bc))
* **lora:** implement adapter dropout (was dead config) ([#51](https://github.com/laurigates/loractl/issues/51)) ([14fa603](https://github.com/laurigates/loractl/commit/14fa603db0f0f3cc8b8a88078a86078d35d7ff0b))


### Bug Fixes

* **trainer:** honor optim.weight_decay via AdamW ([#50](https://github.com/laurigates/loractl/issues/50)) ([594a057](https://github.com/laurigates/loractl/commit/594a057958c4dc09cd3d52dfba888476e0f64269))

## [0.4.1](https://github.com/laurigates/loractl/compare/v0.4.0...v0.4.1) (2026-07-11)


### Bug Fixes

* **deps:** clear security-audit advisories (crossbeam-epoch, num-bigint, indicatif 0.18) ([#35](https://github.com/laurigates/loractl/issues/35)) ([6c98774](https://github.com/laurigates/loractl/commit/6c987746188c9d43440a97e0b6d4659cf9545cfb))

## [0.4.0](https://github.com/laurigates/loractl/compare/v0.3.0...v0.4.0) (2026-07-11)


### Features

* **core:** rectified-flow v-param objective + timestep sampling (M8, [#19](https://github.com/laurigates/loractl/issues/19)) ([#33](https://github.com/laurigates/loractl/issues/33)) ([d859381](https://github.com/laurigates/loractl/commit/d85938152da3155db162b628b2ce4c460eb0ea28))

## [0.3.0](https://github.com/laurigates/loractl/compare/v0.2.0...v0.3.0) (2026-07-09)


### Features

* **core:** generalize LoRA injection + kohya-ss export (M6, [#17](https://github.com/laurigates/loractl/issues/17)) ([#28](https://github.com/laurigates/loractl/issues/28)) ([ce70295](https://github.com/laurigates/loractl/commit/ce70295da01aa81095dd1b45a9c247033378ac79))

## [0.2.0](https://github.com/laurigates/loractl/compare/v0.1.0...v0.2.0) (2026-07-05)


### Features

* **api:** add loractl-api HTTP/SSE event streaming (M5) ([#14](https://github.com/laurigates/loractl/issues/14)) ([fd7896b](https://github.com/laurigates/loractl/commit/fd7896b1e50b707866c42ffafe1237e53433a41e))
* **core:** add burn backend dep and a LoRA adapter module ([#7](https://github.com/laurigates/loractl/issues/7)) ([d6b3188](https://github.com/laurigates/loractl/commit/d6b3188ff688f0edb9e62292a46bf5443b0dde7c))
* **core:** add burn BurnTrainer + LoRA MNIST correctness harness ([#8](https://github.com/laurigates/loractl/issues/8)) ([a08b0d7](https://github.com/laurigates/loractl/commit/a08b0d7f5b4bef3a225325536d1fdacadedde37a))
* **core:** load real GPT-2 weights into burn with forward-pass parity (M3) ([#9](https://github.com/laurigates/loractl/issues/9)) ([f58d328](https://github.com/laurigates/loractl/commit/f58d3283c7503e3505f09e382474ebafc1f966d2))
* **core:** safetensors adapter I/O + sampling (M4) ([#11](https://github.com/laurigates/loractl/issues/11)) ([887cbef](https://github.com/laurigates/loractl/commit/887cbef78edab34ed81d4e93b022b5685e47e2c4))
* report errors to GlitchTip via the Sentry SDK ([#5](https://github.com/laurigates/loractl/issues/5)) ([25c9f27](https://github.com/laurigates/loractl/commit/25c9f27a81e6f1a6ee7d92f02701df20093bab4f))
* scaffold loractl — terminal-native LoRA trainer skeleton ([1f90a71](https://github.com/laurigates/loractl/commit/1f90a7127874dcea9e8a9439b264cb063f7a14d5))
