# Changelog

## [0.13.0](https://github.com/laurigates/loractl/compare/v0.12.0...v0.13.0) (2026-07-19)


### Features

* **core:** block-level gradient checkpointing for the MMDiT trunk ([#134](https://github.com/laurigates/loractl/issues/134)) ([#135](https://github.com/laurigates/loractl/issues/135)) ([a1964af](https://github.com/laurigates/loractl/commit/a1964af5f102d7da070cf9e35c4ead7d2c4d54f4))
* **core:** retention-ledger attribution wiring ([#132](https://github.com/laurigates/loractl/issues/132)) ([b7be0b8](https://github.com/laurigates/loractl/commit/b7be0b898c7ca4c0a1a1e8cf9baf601e9a0cb5fd))
* **quant:** chunk large frozen-base dequants into row chunks ([#128](https://github.com/laurigates/loractl/issues/128)) ([#130](https://github.com/laurigates/loractl/issues/130)) ([5bc9902](https://github.com/laurigates/loractl/commit/5bc9902eec12456f139c89c00be8069c6bd849ae))


### Bug Fixes

* **core:** retention-ledger review fixes — deny allowlist, per-step report ([d112876](https://github.com/laurigates/loractl/commit/d11287604aa3df672c1c6bf73ed08f8429db43ec))

## [0.12.0](https://github.com/laurigates/loractl/compare/v0.11.0...v0.12.0) (2026-07-18)


### Features

* **core:** ADR-0005 step-VRAM probe + memory-lever configs ([#126](https://github.com/laurigates/loractl/issues/126)) ([f5bf4aa](https://github.com/laurigates/loractl/commit/f5bf4aad629106482285f48926e4c4c8f8a26fec))

## [0.11.0](https://github.com/laurigates/loractl/compare/v0.10.0...v0.11.0) (2026-07-18)


### Features

* **bench:** scaffold loractl-bench measurement harness crate ([#115](https://github.com/laurigates/loractl/issues/115)) ([d5fa97b](https://github.com/laurigates/loractl/commit/d5fa97b7002ad8c76c1ab607bfab3157f27b7073))
* **cli:** per-component model path-override flags ([#108](https://github.com/laurigates/loractl/issues/108)) ([42d219b](https://github.com/laurigates/loractl/commit/42d219bd53c5ab64a4cece458f5cbdb4bc765aad))
* **core:** QuantMatmulT-backward GPU diagnostic (quant_grad_compare) ([#114](https://github.com/laurigates/loractl/issues/114)) ([c3f0c76](https://github.com/laurigates/loractl/commit/c3f0c765ace0964d892d577e5dca386fd5739396))
* **flow:** resolution-based timestep shift with Krea 2 endpoints ([#107](https://github.com/laurigates/loractl/issues/107)) ([7ff5ef5](https://github.com/laurigates/loractl/commit/7ff5ef58f6545b4a0c0a092b41e7e9dd172e35d9))
* **quant:** int4 (Q4S) frozen-base quantization to fit a 24 GB step ([#119](https://github.com/laurigates/loractl/issues/119)) ([a2149db](https://github.com/laurigates/loractl/commit/a2149db850aa2bb41f3a171f27390a37f1b6978e))


### Bug Fixes

* **core:** restore example loads with skip_enum_variants ([#117](https://github.com/laurigates/loractl/issues/117)) ([4bc4757](https://github.com/laurigates/loractl/commit/4bc4757c98f65f1776fc1e9390f7b581aab5512f))

## [0.10.0](https://github.com/laurigates/loractl/compare/v0.9.0...v0.10.0) (2026-07-16)


### Features

* **config:** per-component model paths — load scattered ComfyUI files ([#101](https://github.com/laurigates/loractl/issues/101)) ([20a3d7a](https://github.com/laurigates/loractl/commit/20a3d7a68a4a90d4d3eaf419e0087ea7a3b10b37))
* **encode:** load a ComfyUI native-keyed Qwen-Image VAE ([#104](https://github.com/laurigates/loractl/issues/104)) ([ad87df4](https://github.com/laurigates/loractl/commit/ad87df4454476df74b33d8a6c7522f6f3aa5c4f1))
* **encode:** load a ComfyUI-fp8 Qwen3-VL text encoder ([#105](https://github.com/laurigates/loractl/issues/105)) ([7166c64](https://github.com/laurigates/loractl/commit/7166c64edac89bfe4238764f171eadb2e29e8863))
* **hf:** fetch the Qwen3-VL tokenizer on demand for ComfyUI layouts ([#103](https://github.com/laurigates/loractl/issues/103)) ([248d2e3](https://github.com/laurigates/loractl/commit/248d2e30be66ca0211bb3f887c76b550589ca25f))

## [0.9.0](https://github.com/laurigates/loractl/compare/v0.8.0...v0.9.0) (2026-07-16)


### Features

* **api:** optional bearer-token auth gated on LORACTL_API_TOKEN ([#92](https://github.com/laurigates/loractl/issues/92)) ([a00efdd](https://github.com/laurigates/loractl/commit/a00efddf93f2982894d35d48fd0efcaa24e5d839))
* **core:** cuda arms for the numerics ladder — grad_compare + cuda smoke ([#94](https://github.com/laurigates/loractl/issues/94)) ([002d421](https://github.com/laurigates/loractl/commit/002d421f5267cacb47a13fc42de17863bd91207f))
* **mmdit:** quantizable BaseLinear sites (int8 frozen base) ([#98](https://github.com/laurigates/loractl/issues/98)) ([8ccc40a](https://github.com/laurigates/loractl/commit/8ccc40a16c15071ea4cb48048c7d6d1210c2d502))
* **quant:** int8 QuantBackend core — scheme, custom autodiff op, goldens ([#97](https://github.com/laurigates/loractl/issues/97)) ([69b546b](https://github.com/laurigates/loractl/commit/69b546b65a2613232fd023938af35e677a5059c5))
* **quant:** on-box int8 validation — quant_probe + cuda int8 e2e ([#100](https://github.com/laurigates/loractl/issues/100)) ([6fda532](https://github.com/laurigates/loractl/commit/6fda532db8b66d62c443ecc47c16a4bf46964862))
* **trainer:** compute.quant=int8 — quantized frozen-base training ([#99](https://github.com/laurigates/loractl/issues/99)) ([fb0f3a8](https://github.com/laurigates/loractl/commit/fb0f3a8ac0001c75d713d74fa7f13b2000f77079))
* **trainer:** cuda backend for DiffusionTrainer (f32-only) ([#95](https://github.com/laurigates/loractl/issues/95)) ([562eb37](https://github.com/laurigates/loractl/commit/562eb3754374824678c5ebe52824f730f6cf7bec))

## [0.8.0](https://github.com/laurigates/loractl/compare/v0.7.0...v0.8.0) (2026-07-15)


### Features

* **cli:** add `loractl init` — scaffold a config from a template ([#89](https://github.com/laurigates/loractl/issues/89)) ([7e3a19c](https://github.com/laurigates/loractl/commit/7e3a19c871870d77ee8eb3566dc5950de6465dbf))

## [0.7.0](https://github.com/laurigates/loractl/compare/v0.6.0...v0.7.0) (2026-07-15)


### Features

* **trainer:** train on Krea-2-Turbo — variant seam + scaled-fp8 checkpoint loader (M15) ([#85](https://github.com/laurigates/loractl/issues/85)) ([856d85f](https://github.com/laurigates/loractl/commit/856d85fdf8f275e46713b7b210543f28f3c391a1))

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
