/// Integration test for neural network ops:
/// linear, embedding, layer_norm, rms_norm, dropout, batch_norm,
/// attention, mse_loss, cross_entropy_loss, bce_loss,
/// conv2d, max_pool2d, avg_pool2d, adaptive_avg_pool2d,
/// SGD, Adam.
use std::sync::Arc;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    if a.is_infinite() && b.is_infinite() { return a.signum() == b.signum(); }
    (a - b).abs() < tol
}

fn main() {
    let config = ptx_runtime::PTXStableConfig {
        struct_size: std::mem::size_of::<ptx_runtime::PTXStableConfig>() as u32,
        abi_version: ptx_runtime::PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.50,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: 16,
        quiet_init: 1,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = Arc::new(
        ptx_runtime::PtxRuntime::with_stable_config(0, Some(config))
            .expect("runtime init failed"),
    );
    runtime.export_for_hook();
    runtime.export_context();

    let mut pass = 0u32;
    let mut fail = 0u32;
    let tol = 1e-3;

    macro_rules! tensor {
        ($data:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data,
                &[$data.len()],
                ptx_tensor::DType::F32,
                &runtime,
            )
            .unwrap()
        };
    }

    macro_rules! tensor2d {
        ($data:expr, $rows:expr, $cols:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data,
                &[$rows, $cols],
                ptx_tensor::DType::F32,
                &runtime,
            )
            .unwrap()
        };
    }

    macro_rules! tensor_nd {
        ($data:expr, $shape:expr) => {
            ptx_tensor::Tensor::from_slice(
                $data,
                $shape,
                ptx_tensor::DType::F32,
                &runtime,
            )
            .unwrap()
        };
    }

    macro_rules! check {
        ($name:expr, $result:expr, $expected:expr) => {{
            let res = $result.to_vec_f32().unwrap();
            let exp: &[f32] = $expected;
            let ok = res.len() == exp.len()
                && res.iter().zip(exp.iter()).all(|(a, b)| approx_eq(*a, *b, tol));
            if ok {
                println!("PASS: {}", $name);
                pass += 1;
            } else {
                println!(
                    "FAIL: {} — expected {:?}, got {:?}",
                    $name,
                    &exp[..exp.len().min(16)],
                    &res[..res.len().min(16)]
                );
                fail += 1;
            }
        }};
    }

    macro_rules! check_shape {
        ($name:expr, $tensor:expr, $expected_shape:expr) => {{
            let shape = $tensor.shape();
            let exp: &[usize] = $expected_shape;
            if shape == exp {
                println!("PASS: {} shape {:?}", $name, shape);
                pass += 1;
            } else {
                println!("FAIL: {} shape — expected {:?}, got {:?}", $name, exp, shape);
                fail += 1;
            }
        }};
    }

    println!("=== Neural Network Ops Tests ===\n");

    // =====================================================================
    // 1. Linear
    // =====================================================================
    println!("--- linear ---");
    {
        // x: (2, 3), weight: (4, 3) → out: (2, 4)
        let x = tensor2d!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let w = tensor2d!(
            &[
                1.0f32, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
            ],
            4, 3
        );
        let bias = tensor!(&[0.1f32, 0.2, 0.3, 0.4]);
        let out = x.linear(&w, Some(&bias)).unwrap();
        check_shape!("linear shape", out, &[2, 4]);
        // Row 0: [1,2,3] @ eye columns + [1,1,1] col = [1,2,3,6] + bias
        // Row 1: [4,5,6] @ eye columns + [1,1,1] col = [4,5,6,15] + bias
        check!("linear values", out, &[1.1, 2.2, 3.3, 6.4, 4.1, 5.2, 6.3, 15.4]);
    }
    {
        // Linear without bias
        let x = tensor2d!(&[1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let w = tensor2d!(&[2.0f32, 3.0, 4.0, 5.0], 2, 2);
        let out = x.linear(&w, None).unwrap();
        // x @ w.T = [[1,0],[0,1]] @ [[2,4],[3,5]] = [[2,4],[3,5]]
        check!("linear no bias", out, &[2.0, 4.0, 3.0, 5.0]);
    }

    // =====================================================================
    // 2. Embedding
    // =====================================================================
    println!("\n--- embedding ---");
    {
        // weight: (5, 3) embedding table
        let weight = tensor2d!(
            &[
                0.0f32, 0.1, 0.2,   // token 0
                1.0, 1.1, 1.2,      // token 1
                2.0, 2.1, 2.2,      // token 2
                3.0, 3.1, 3.2,      // token 3
                4.0, 4.1, 4.2,      // token 4
            ],
            5, 3
        );
        let indices = ptx_tensor::Tensor::from_slice(
            &[1i32, 3, 0, 4],
            &[4],
            ptx_tensor::DType::I32,
            &runtime,
        ).unwrap();
        let out = ptx_tensor::Tensor::embedding(&weight, &indices).unwrap();
        check_shape!("embedding shape", out, &[4, 3]);
        check!("embedding values", out, &[1.0, 1.1, 1.2, 3.0, 3.1, 3.2, 0.0, 0.1, 0.2, 4.0, 4.1, 4.2]);
    }

    // =====================================================================
    // 3. Layer Norm
    // =====================================================================
    println!("\n--- layer_norm ---");
    {
        // Simple: normalize a (2, 4) tensor over last dim
        let x = tensor2d!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
        let out = x.layer_norm(1, None, None, 1e-5).unwrap();
        check_shape!("layer_norm shape", out, &[2, 4]);
        // Each row should be normalized: mean=0, var≈1
        let vals = out.to_vec_f32().unwrap();
        let row0_mean: f32 = vals[0..4].iter().sum::<f32>() / 4.0;
        let row0_var: f32 = vals[0..4].iter().map(|v| (v - row0_mean).powi(2)).sum::<f32>() / 4.0;
        if row0_mean.abs() < 0.01 && (row0_var - 1.0).abs() < 0.1 {
            println!("PASS: layer_norm stats (mean≈0, var≈1)");
            pass += 1;
        } else {
            println!("FAIL: layer_norm stats — mean={}, var={}", row0_mean, row0_var);
            fail += 1;
        }
    }
    {
        // With weight and bias
        let x = tensor2d!(&[1.0f32, 2.0, 3.0, 4.0], 1, 4);
        let w = tensor!(&[2.0f32, 2.0, 2.0, 2.0]);
        let b = tensor!(&[1.0f32, 1.0, 1.0, 1.0]);
        let out = x.layer_norm(1, Some(&w), Some(&b), 1e-5).unwrap();
        check_shape!("layer_norm+affine shape", out, &[1, 4]);
        // Should be scaled and shifted
        let vals = out.to_vec_f32().unwrap();
        let mean: f32 = vals.iter().sum::<f32>() / 4.0;
        if (mean - 1.0).abs() < 0.1 {
            println!("PASS: layer_norm+affine mean≈1");
            pass += 1;
        } else {
            println!("FAIL: layer_norm+affine mean={}", mean);
            fail += 1;
        }
    }

    // =====================================================================
    // 4. RMS Norm
    // =====================================================================
    println!("\n--- rms_norm ---");
    {
        let x = tensor2d!(&[3.0f32, 4.0], 1, 2);
        // rms = sqrt((9+16)/2) = sqrt(12.5) = 3.5355
        // out = [3/3.5355, 4/3.5355] = [0.8485, 1.1314]
        let out = x.rms_norm(1, None, 1e-5).unwrap();
        check!("rms_norm", out, &[0.8485, 1.1314]);
    }

    // =====================================================================
    // 5. Dropout
    // =====================================================================
    println!("\n--- dropout ---");
    {
        let x = ptx_tensor::Tensor::ones(&[1000], ptx_tensor::DType::F32, &runtime).unwrap();

        // Inference mode: should be identity
        let (out, mask) = x.dropout(0.5, false).unwrap();
        let vals = out.to_vec_f32().unwrap();
        let all_ones = vals.iter().all(|v| approx_eq(*v, 1.0, tol));
        if all_ones && mask.is_none() {
            println!("PASS: dropout inference (identity)");
            pass += 1;
        } else {
            println!("FAIL: dropout inference");
            fail += 1;
        }

        // Training mode: ~50% should be zero, rest scaled by 2
        let (out2, mask2) = x.dropout(0.5, true).unwrap();
        let vals2 = out2.to_vec_f32().unwrap();
        let zeros = vals2.iter().filter(|v| **v == 0.0).count();
        let twos = vals2.iter().filter(|v| approx_eq(**v, 2.0, 0.01)).count();
        let ratio = zeros as f32 / 1000.0;
        if ratio > 0.3 && ratio < 0.7 && mask2.is_some() && (zeros + twos) == 1000 {
            println!("PASS: dropout training (ratio={:.2}, zeros={}, twos={})", ratio, zeros, twos);
            pass += 1;
        } else {
            println!("FAIL: dropout training — ratio={:.2}, zeros={}, twos={}", ratio, zeros, twos);
            fail += 1;
        }
    }

    // =====================================================================
    // 6. Batch Norm (inference)
    // =====================================================================
    println!("\n--- batch_norm ---");
    {
        // Input: (1, 2, 2, 2) — 1 batch, 2 channels, 2x2 spatial
        let x = tensor_nd!(
            &[
                1.0f32, 2.0, 3.0, 4.0,  // channel 0
                5.0, 6.0, 7.0, 8.0,     // channel 1
            ],
            &[1, 2, 2, 2]
        );
        let mean = tensor!(&[2.5f32, 6.5]);  // per-channel mean
        let var = tensor!(&[1.25f32, 1.25]); // per-channel var
        let out = x.batch_norm(&mean, &var, None, None, 1e-5).unwrap();
        check_shape!("batch_norm shape", out, &[1, 2, 2, 2]);
        // Channel 0: (x - 2.5) / sqrt(1.25 + eps) ≈ (x - 2.5) / 1.1180
        // [-1.3416, -0.4472, 0.4472, 1.3416]
        let vals = out.to_vec_f32().unwrap();
        if approx_eq(vals[0], -1.3416, 0.01) && approx_eq(vals[3], 1.3416, 0.01) {
            println!("PASS: batch_norm values");
            pass += 1;
        } else {
            println!("FAIL: batch_norm values — got {:?}", &vals[..8]);
            fail += 1;
        }
    }

    // =====================================================================
    // 7. MSE Loss
    // =====================================================================
    println!("\n--- mse_loss ---");
    {
        let pred = tensor!(&[1.0f32, 2.0, 3.0]);
        let target = tensor!(&[1.5f32, 2.5, 3.5]);
        let loss = pred.mse_loss(&target, ptx_tensor::Reduction::Mean).unwrap();
        // ((0.5^2 + 0.5^2 + 0.5^2) / 3) = 0.25
        check!("mse_loss mean", loss, &[0.25]);
    }
    {
        let pred = tensor!(&[1.0f32, 2.0]);
        let target = tensor!(&[3.0f32, 5.0]);
        let loss = pred.mse_loss(&target, ptx_tensor::Reduction::Sum).unwrap();
        // (4 + 9) = 13
        check!("mse_loss sum", loss, &[13.0]);
    }
    {
        let pred = tensor!(&[1.0f32, 2.0]);
        let target = tensor!(&[1.0f32, 3.0]);
        let loss = pred.mse_loss(&target, ptx_tensor::Reduction::None).unwrap();
        check!("mse_loss none", loss, &[0.0, 1.0]);
    }

    // =====================================================================
    // 8. Cross-Entropy Loss
    // =====================================================================
    println!("\n--- cross_entropy_loss ---");
    {
        // logits: (2, 3), targets: (2,) class indices
        let logits = tensor2d!(&[2.0f32, 1.0, 0.1, 0.1, 1.0, 2.0], 2, 3);
        let targets = ptx_tensor::Tensor::from_slice(
            &[0i32, 2],
            &[2],
            ptx_tensor::DType::I32,
            &runtime,
        ).unwrap();
        let loss = logits.cross_entropy_loss(&targets, ptx_tensor::Reduction::Mean).unwrap();
        let val = loss.to_vec_f32().unwrap()[0];
        // log_softmax of [2,1,0.1] at idx 0: log(exp(2)/sum) ≈ -0.4170
        // log_softmax of [0.1,1,2] at idx 2: log(exp(2)/sum) ≈ -0.4170
        // mean nll ≈ 0.4170
        if val > 0.3 && val < 0.6 {
            println!("PASS: cross_entropy_loss (val={:.4})", val);
            pass += 1;
        } else {
            println!("FAIL: cross_entropy_loss — expected ~0.42, got {:.4}", val);
            fail += 1;
        }
    }

    // =====================================================================
    // 9. Binary Cross-Entropy Loss
    // =====================================================================
    println!("\n--- binary_cross_entropy_loss ---");
    {
        let pred = tensor!(&[0.9f32, 0.1, 0.8]);
        let target = tensor!(&[1.0f32, 0.0, 1.0]);
        let loss = pred.binary_cross_entropy_loss(&target, ptx_tensor::Reduction::Mean).unwrap();
        let val = loss.to_vec_f32().unwrap()[0];
        // BCE = -(1*log(0.9) + 0*log(0.1) + 1*log(0.8)) / 3
        // + -(0*log(0.1) + 1*log(0.9) + 0*log(0.2)) / 3
        // ≈ 0.1456
        if val > 0.05 && val < 0.3 {
            println!("PASS: bce_loss (val={:.4})", val);
            pass += 1;
        } else {
            println!("FAIL: bce_loss — expected ~0.15, got {:.4}", val);
            fail += 1;
        }
    }

    // =====================================================================
    // 10. Scaled Dot-Product Attention
    // =====================================================================
    println!("\n--- scaled_dot_product_attention ---");
    {
        // Q, K, V: (1, 1, 2, 3) — batch=1, heads=1, seq=2, dim=3
        let q = tensor_nd!(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[1, 1, 2, 3]);
        let k = tensor_nd!(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[1, 1, 2, 3]);
        let v = tensor_nd!(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3]);

        let out = ptx_tensor::Tensor::scaled_dot_product_attention(
            &q, &k, &v, None, 0.0, false,
        ).unwrap();
        check_shape!("attention shape", out, &[1, 1, 2, 3]);

        // Q[0]=[1,0,0], K^T columns=[1,0;0,1;0,0]
        // attn_scores[0] = [1/sqrt(3), 0/sqrt(3)] → softmax → [~0.65, ~0.35]
        // output ≈ weighted sum of V rows
        let vals = out.to_vec_f32().unwrap();
        // Just check the output is reasonable (between V row values)
        if vals[0] > 0.5 && vals[0] < 4.5 && vals.len() == 6 {
            println!("PASS: attention values reasonable");
            pass += 1;
        } else {
            println!("FAIL: attention values — got {:?}", vals);
            fail += 1;
        }
    }
    {
        // Causal attention
        let q = tensor_nd!(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let k = tensor_nd!(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let v = tensor_nd!(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);

        let out = ptx_tensor::Tensor::scaled_dot_product_attention(
            &q, &k, &v, None, 0.0, true,
        ).unwrap();
        check_shape!("causal attention shape", out, &[1, 1, 2, 2]);
        // First position can only attend to itself, second to both
        let vals = out.to_vec_f32().unwrap();
        // Position 0 attends only to pos 0: out = v[0] = [1, 0]
        if approx_eq(vals[0], 1.0, 0.01) && approx_eq(vals[1], 0.0, 0.01) {
            println!("PASS: causal attention pos0 = v[0]");
            pass += 1;
        } else {
            println!("FAIL: causal attention pos0 — got [{}, {}]", vals[0], vals[1]);
            fail += 1;
        }
    }

    // =====================================================================
    // 11. Conv2d
    // =====================================================================
    println!("\n--- conv2d ---");
    {
        // Input: (1, 1, 4, 4), Weight: (1, 1, 3, 3) identity-like kernel
        #[rustfmt::skip]
        let input_data = [
            1.0f32, 2.0, 3.0, 0.0,
            4.0, 5.0, 6.0, 0.0,
            7.0, 8.0, 9.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let x = tensor_nd!(&input_data, &[1, 1, 4, 4]);

        // Simple averaging kernel
        let w_data = [1.0f32 / 9.0; 9];
        let w = tensor_nd!(&w_data, &[1, 1, 3, 3]);

        let out = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1]).unwrap();
        check_shape!("conv2d shape (no pad)", out, &[1, 1, 2, 2]);
        // Center of 3x3 average
        let vals = out.to_vec_f32().unwrap();
        // Top-left 3x3: avg(1,2,3,4,5,6,7,8,9) = 5.0
        if approx_eq(vals[0], 5.0, 0.01) {
            println!("PASS: conv2d center value = 5.0");
            pass += 1;
        } else {
            println!("FAIL: conv2d center value — expected 5.0, got {}", vals[0]);
            fail += 1;
        }
    }
    {
        // Conv2d with padding
        let x = tensor_nd!(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        // Identity 1x1 conv
        let w = tensor_nd!(&[1.0f32], &[1, 1, 1, 1]);
        let out = x.conv2d(&w, None, [1, 1], [0, 0], [1, 1]).unwrap();
        check_shape!("conv2d 1x1", out, &[1, 1, 2, 2]);
        check!("conv2d 1x1 values", out, &[1.0, 2.0, 3.0, 4.0]);
    }
    {
        // Conv2d with bias
        let x = tensor_nd!(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let w = tensor_nd!(&[1.0f32], &[1, 1, 1, 1]);
        let b = tensor!(&[10.0f32]);
        let out = x.conv2d(&w, Some(&b), [1, 1], [0, 0], [1, 1]).unwrap();
        check!("conv2d 1x1+bias", out, &[11.0, 12.0, 13.0, 14.0]);
    }

    // =====================================================================
    // 12. MaxPool2d
    // =====================================================================
    println!("\n--- max_pool2d ---");
    {
        #[rustfmt::skip]
        let x = tensor_nd!(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            &[1, 1, 4, 4]
        );
        let out = x.max_pool2d([2, 2], [2, 2], [0, 0]).unwrap();
        check_shape!("max_pool2d shape", out, &[1, 1, 2, 2]);
        check!("max_pool2d values", out, &[6.0, 8.0, 14.0, 16.0]);
    }

    // =====================================================================
    // 13. AvgPool2d
    // =====================================================================
    println!("\n--- avg_pool2d ---");
    {
        #[rustfmt::skip]
        let x = tensor_nd!(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            &[1, 1, 4, 4]
        );
        let out = x.avg_pool2d([2, 2], [2, 2], [0, 0]).unwrap();
        check_shape!("avg_pool2d shape", out, &[1, 1, 2, 2]);
        // avg of [1,2,5,6]=3.5, [3,4,7,8]=5.5, [9,10,13,14]=11.5, [11,12,15,16]=13.5
        check!("avg_pool2d values", out, &[3.5, 5.5, 11.5, 13.5]);
    }

    // =====================================================================
    // 14. Adaptive Avg Pool2d
    // =====================================================================
    println!("\n--- adaptive_avg_pool2d ---");
    {
        #[rustfmt::skip]
        let x = tensor_nd!(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            &[1, 1, 4, 4]
        );
        let out = x.adaptive_avg_pool2d([1, 1]).unwrap();
        check_shape!("adaptive_avg_pool2d 1x1 shape", out, &[1, 1, 1, 1]);
        // Global average = (1+2+...+16)/16 = 8.5
        check!("adaptive_avg_pool2d 1x1 value", out, &[8.5]);
    }

    // =====================================================================
    // 15. SGD Optimizer
    // =====================================================================
    println!("\n--- SGD ---");
    {
        let mut params = vec![
            ptx_tensor::Tensor::full(&[4], 5.0, ptx_tensor::DType::F32, &runtime).unwrap()
        ];
        let grads = vec![
            ptx_tensor::Tensor::ones(&[4], ptx_tensor::DType::F32, &runtime).unwrap()
        ];
        let mut sgd = ptx_tensor::ops::optim::SGD::new(0.1, 0.0, 0.0);
        sgd.step(&mut params, &grads).unwrap();
        // params -= lr * grads → 5.0 - 0.1 * 1.0 = 4.9
        check!("sgd step", params[0], &[4.9, 4.9, 4.9, 4.9]);
    }

    // =====================================================================
    // 16. Adam Optimizer
    // =====================================================================
    println!("\n--- Adam ---");
    {
        let mut params = vec![
            ptx_tensor::Tensor::full(&[4], 5.0, ptx_tensor::DType::F32, &runtime).unwrap()
        ];
        let grads = vec![
            ptx_tensor::Tensor::ones(&[4], ptx_tensor::DType::F32, &runtime).unwrap()
        ];
        let mut adam = ptx_tensor::ops::optim::Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        adam.step(&mut params, &grads).unwrap();
        // After 1 step with gradient=1, params should have decreased slightly
        let vals = params[0].to_vec_f32().unwrap();
        if vals[0] < 5.0 && vals[0] > 4.99 {
            println!("PASS: adam step (val={:.6})", vals[0]);
            pass += 1;
        } else {
            println!("FAIL: adam step — expected ~4.999, got {:.6}", vals[0]);
            fail += 1;
        }
    }

    // =====================================================================
    // Summary
    // =====================================================================
    println!("\n=== NN Ops Results: {} passed, {} failed ===", pass, fail);
    if fail > 0 {
        std::process::exit(1);
    }
}
