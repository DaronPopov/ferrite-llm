// Phi-2 Quantized Inference (4-bit GGUF)
//
// Runs Phi-2 with 4-bit quantization using GGUF format.
// Requires ~2GB VRAM instead of ~11GB for F32.
//
// Custom loader for GGUF files with separate Q/K/V tensors.

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm};
use candle_transformers::generation::LogitsProcessor;
use ferrite::Tokenizer;
use std::collections::HashMap;
use std::io::{self, Read, Seek, Write};
use std::path::PathBuf;
use std::time::Instant;

const MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone)]
struct QLinear {
    inner: QMatMul,
    bias: Tensor,
}

impl QLinear {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let w = ct.tensor(r, &format!("{name}.weight"), device)?;
        let b = ct.tensor(r, &format!("{name}.bias"), device)?;
        let inner = QMatMul::from_qtensor(w)?;
        let bias = b.dequantize(device)?;
        Ok(Self { inner, bias })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)?.broadcast_add(&self.bias)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    ffn_up: QLinear,
    ffn_down: QLinear,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.ffn_up)?.gelu()?.apply(&self.ffn_down)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_q: QLinear,
    attn_k: QLinear,
    attn_v: QLinear,
    attn_output: QLinear,
    attn_norm: LayerNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    rope_dim: usize,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, xs: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _n_embd) = xs.dims4()?;
        let xs_rot = xs.i((.., .., .., ..self.rope_dim))?;
        let xs_pass = xs.i((.., .., .., self.rope_dim..))?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let xs_rot = candle_nn::rotary_emb::rope(&xs_rot.contiguous()?, &cos, &sin)?;
        Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        // Separate Q, K, V projections
        let q = self.attn_q.forward(x)?;
        let k = self.attn_k.forward(x)?;
        let v = self.attn_v.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?.contiguous()?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k.contiguous()?, v.contiguous()?),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k.contiguous()?, v.contiguous()?)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k.contiguous()?, v.contiguous()?)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA if needed
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.attn_output.forward(&y)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: LayerNorm,
    output: QLinear,
    masks: HashMap<usize, Tensor>,
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

fn layer_norm(w: QTensor, b: QTensor, eps: f64) -> Result<LayerNorm> {
    let w = w.dequantize(&w.device())?;
    let b = b.dequantize(&b.device())?;
    Ok(LayerNorm::new(w, b, eps))
}

impl ModelWeights {
    pub fn from_gguf<R: Seek + Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("phi2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("phi2.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("phi2.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("phi2.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("phi2.rope.dimension_count")?.to_u32()? as usize;
        let ln_eps = md_get("phi2.attention.layer_norm_epsilon")?.to_f32()? as f64;

        let (cos, sin) = precompute_freqs_cis(rope_dim, 10_000., device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;

        let output_norm = layer_norm(
            ct.tensor(reader, "output_norm.weight", device)?,
            ct.tensor(reader, "output_norm.bias", device)?,
            ln_eps,
        )?;
        let output = QLinear::new(&ct, reader, "output", device)?;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let ffn_up = QLinear::new(&ct, reader, &format!("{prefix}.ffn_up"), device)?;
            let ffn_down = QLinear::new(&ct, reader, &format!("{prefix}.ffn_down"), device)?;
            let mlp = Mlp { ffn_up, ffn_down };

            let attn_norm = layer_norm(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                ct.tensor(reader, &format!("{prefix}.attn_norm.bias"), device)?,
                ln_eps,
            )?;

            layers.push(LayerWeights {
                attn_q: QLinear::new(&ct, reader, &format!("{prefix}.attn_q"), device)?,
                attn_k: QLinear::new(&ct, reader, &format!("{prefix}.attn_k"), device)?,
                attn_v: QLinear::new(&ct, reader, &format!("{prefix}.attn_v"), device)?,
                attn_output: QLinear::new(&ct, reader, &format!("{prefix}.attn_output"), device)?,
                attn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                rope_dim,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            })
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            masks: HashMap::new(),
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, xs: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = xs.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, xs.device())?)
        };

        let mut xs = self.tok_embeddings.forward(xs)?;
        for layer in self.layers.iter_mut() {
            let residual = &xs;
            let xs_norm = xs.apply(&layer.attn_norm)?;
            let attn_outputs = layer.forward_attn(&xs_norm, mask.as_ref(), index_pos)?;
            let feed_forward_hidden_states = layer.mlp.forward(&xs_norm)?;
            xs = (attn_outputs + feed_forward_hidden_states + residual)?
        }

        let xs = xs.apply(&self.output_norm)?.i((.., seq_len - 1, ..))?;
        self.output.forward(&xs)
    }
}

struct QuantizedGenerator {
    model: ModelWeights,
    device: Device,
    tokenizer: Tokenizer,
}

impl QuantizedGenerator {
    fn new(
        gguf_path: &PathBuf,
        tokenizer_dir: &PathBuf,
        device: Device,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        println!("[Init] Loading tokenizer...");
        let tokenizer = Tokenizer::from_dir(tokenizer_dir)?;
        println!("[Init] Vocab size: {}", tokenizer.vocab_size());

        println!("[Init] Loading quantized GGUF model...");
        let mut file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(gguf_content, &mut file, &device)?;

        println!("[Init] Model ready! (4-bit quantized)");

        Ok(Self {
            model,
            device,
            tokenizer,
        })
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> std::result::Result<String, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(prompt)?;
        let tokens: Vec<u32> = encoding.ids.clone();
        let prompt_len = tokens.len();

        println!("[Generate] Prompt: {} tokens", prompt_len);

        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));
        let mut decoder = self.tokenizer.decode_stream(&tokens, true);
        let mut all_tokens = tokens.clone();

        // Process prompt tokens one by one for quantized model
        let mut pos = 0;
        for &token in &tokens {
            let input = Tensor::new(&[[token]], &self.device)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }

        // Get logits for last position
        let last_token = *tokens.last().unwrap();
        let input = Tensor::new(&[[last_token]], &self.device)?;
        let logits = self.model.forward(&input, pos - 1)?;
        let logits = logits.squeeze(0)?;

        let mut next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if let Ok(Some(text)) = decoder.step(next_token) {
            print!("{}", text);
            io::stdout().flush()?;
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(50256);

        let decode_start = Instant::now();
        let mut generated_tokens = 1usize;

        for _ in 0..max_tokens - 1 {
            if next_token == eos_id {
                break;
            }

            let input = Tensor::new(&[[next_token]], &self.device)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            pos += 1;

            next_token = logits_processor.sample(&logits)?;

            if next_token == eos_id {
                break;
            }

            all_tokens.push(next_token);
            generated_tokens += 1;

            if let Ok(Some(text)) = decoder.step(next_token) {
                print!("{}", text);
                io::stdout().flush()?;
            }
        }

        if let Ok(Some(text)) = decoder.flush() {
            print!("{}", text);
        }
        println!();

        let elapsed = decode_start.elapsed();
        let tps = generated_tokens as f64 / elapsed.as_secs_f64();
        println!(
            "[Stats] {} tokens in {:.2}s ({:.1} tok/s)",
            generated_tokens,
            elapsed.as_secs_f64(),
            tps
        );

        self.tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| e.into())
    }
}

async fn download_gguf(repo_id: &str, filename: &str) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("huggingface")
        .join("gguf");

    std::fs::create_dir_all(&cache_dir)?;

    let local_path = cache_dir.join(filename);

    if local_path.exists() {
        println!("[Download] Using cached: {:?}", local_path);
        return Ok(local_path);
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );

    println!("[Download] Fetching {} ...", filename);
    println!("[Download] URL: {}", url);

    let response = reqwest::get(&url).await?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let bytes = response.bytes().await?;
    std::fs::write(&local_path, &bytes)?;

    println!("[Download] Saved to: {:?}", local_path);
    Ok(local_path)
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     PHI-2 QUANTIZED - 4-bit GGUF                             ║");
    println!("║     ~2GB VRAM instead of ~11GB                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;
    match &device {
        Device::Cuda(_) => println!("[Init] Using CUDA"),
        Device::Cpu => println!("[Init] Using CPU"),
        _ => {}
    }

    // Download quantized GGUF model (MaziyarPanahi has proper phi2.* metadata keys)
    let gguf_repo = "MaziyarPanahi/phi-2-GGUF";
    let gguf_file = "phi-2.Q4_K_M.gguf";

    println!("\n[Download] Getting quantized model...");
    let gguf_path = download_gguf(gguf_repo, gguf_file).await?;

    // Get tokenizer from original model
    let tokenizer_repo = "microsoft/phi-2";
    println!("[Download] Getting tokenizer...");
    let tokenizer_dir = llm_tokenizer::hub::from_hf(tokenizer_repo, false).await?;

    println!();

    let mut generator = QuantizedGenerator::new(&gguf_path, &tokenizer_dir, device)?;

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  Ready! Type a prompt and press Enter. 'quit' to exit.");
    println!("  Phi-2 works best with code/reasoning prompts.");
    println!("════════════════════════════════════════════════════════════════\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }

        if input.is_empty() {
            continue;
        }

        let prompt = format!("Instruct: {}\nOutput:", input);

        print!("Phi-2-Q4: ");
        io::stdout().flush()?;

        if let Err(e) = generator.generate(&prompt, 1024, 0.7, 0.9) {
            println!("Error: {}", e);
        }
    }

    println!("\nGoodbye!");
    Ok(())
}
