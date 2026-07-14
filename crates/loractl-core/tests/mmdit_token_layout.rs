//! Element-order pins for the two token-layout helpers the diffusion
//! trainer feeds the MMDiT with (review: this seam was previously covered
//! only by doc comments — `mmdit_parity.rs` feeds pre-built tokens from the
//! golden, and every e2e assertion is invariant under a deterministic-but-
//! wrong permutation, so a scrambled layout would train "successfully" and
//! only diverge from real Krea 2 at interop time).
//!
//! The expected values are derived by independent index arithmetic (the
//! same semantics as `sampling.py`'s einops strings, expressed without
//! reshape/permute), plus hand-computed literals for human-readable spots.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use loractl_core::mmdit::{krea2_positions, patchify};

type TB = NdArray;

/// `patchify` must be `sampling.py`'s
/// `rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`: tokens in row-major
/// patch-grid order, each token channel-major (`c`, then `ph`, then `pw`).
#[test]
fn patchify_matches_the_channel_major_rearrange() {
    let device = Default::default();
    let (b, c, h, w, p) = (2usize, 2usize, 4usize, 6usize, 2usize);
    let (gh, gw) = (h / p, w / p);

    // arange input: value == flat index, so every output element identifies
    // exactly which input element landed there.
    let n = b * c * h * w;
    let input = Tensor::<TB, 4>::from_data(
        TensorData::new((0..n).map(|i| i as f32).collect::<Vec<_>>(), [b, c, h, w]),
        &device,
    );

    let out = patchify(input, p);
    assert_eq!(out.dims(), [b, gh * gw, c * p * p]);
    let out: Vec<f32> = out.into_data().convert::<f32>().into_vec().unwrap();

    // Independent derivation: out[bi][r*gw + col][ch*p*p + pr*p + pc]
    //   == input[bi][ch][r*p + pr][col*p + pc], with input value = flat index.
    for bi in 0..b {
        for r in 0..gh {
            for col in 0..gw {
                for ch in 0..c {
                    for pr in 0..p {
                        for pc in 0..p {
                            let token = r * gw + col;
                            let k = ch * p * p + pr * p + pc;
                            let got = out[(bi * gh * gw + token) * (c * p * p) + k];
                            let expect =
                                (bi * c * h * w + ch * h * w + (r * p + pr) * w + (col * p + pc))
                                    as f32;
                            assert_eq!(
                                got, expect,
                                "token ({r},{col}) k={k} (ch={ch},pr={pr},pc={pc}) of batch {bi}"
                            );
                        }
                    }
                }
            }
        }
    }

    // Hand-computed literals (batch 0): token 0 = patch (0,0) — channel 0's
    // 2×2 block, then channel 1's; token 3 = patch (1,0), i.e. input rows 2–3.
    assert_eq!(&out[..8], &[0.0, 1.0, 6.0, 7.0, 24.0, 25.0, 30.0, 31.0][..]);
    assert_eq!(
        &out[3 * 8..4 * 8],
        &[12.0, 13.0, 18.0, 19.0, 36.0, 37.0, 42.0, 43.0][..]
    );
}

/// `krea2_positions` must be `sampling.py`'s `prepare()`: text tokens all at
/// the origin `(0, 0, 0)`, image tokens at `(0, row, col)` in row-major
/// patch-grid order, identical across the batch.
#[test]
fn krea2_positions_match_the_prepare_grid() {
    let device = Default::default();
    let (txt_len, gh, gw, batch) = (3usize, 2usize, 3usize, 2usize);

    let pos = krea2_positions::<TB>(txt_len, gh, gw, batch, &device);
    assert_eq!(pos.dims(), [batch, txt_len + gh * gw, 3]);
    let pos: Vec<f32> = pos.into_data().convert::<f32>().into_vec().unwrap();

    #[rustfmt::skip]
    let one_batch = [
        0.0, 0.0, 0.0, // text
        0.0, 0.0, 0.0, // text
        0.0, 0.0, 0.0, // text
        0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 2.0, // image row 0
        0.0, 1.0, 0.0,  0.0, 1.0, 1.0,  0.0, 1.0, 2.0, // image row 1
    ];
    assert_eq!(&pos[..one_batch.len()], &one_batch[..], "batch 0");
    assert_eq!(&pos[one_batch.len()..], &one_batch[..], "batch 1 identical");
}
