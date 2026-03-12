use anyhow::{anyhow, Result};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CvValueId(usize);

impl CvValueId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub struct Conv2dSpec {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: [i64; 2],
    pub padding: [i64; 2],
    pub dilation: [i64; 2],
    pub groups: i64,
}

#[derive(Debug)]
pub struct Conv2dCfg {
    pub bias: Option<Tensor>,
    pub stride: [i64; 2],
    pub padding: [i64; 2],
    pub dilation: [i64; 2],
    pub groups: i64,
}

impl Conv2dCfg {
    pub fn same() -> Self {
        Self {
            bias: None,
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }

    pub fn with_bias(mut self, bias: Tensor) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn stride(mut self, stride: [i64; 2]) -> Self {
        self.stride = stride;
        self
    }

    pub fn padding(mut self, padding: [i64; 2]) -> Self {
        self.padding = padding;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Upsample2dSpec {
    pub scale_h: i64,
    pub scale_w: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct YoloDecodeSpec {
    pub input_is_xywh: bool,
    pub score_threshold: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct NmsSpec {
    pub iou_threshold: f64,
    pub score_threshold: f64,
    pub max_detections: usize,
    pub class_aware: bool,
}

#[derive(Debug)]
pub enum CvOp {
    Input,
    Conv2d {
        src: CvValueId,
        spec: Conv2dSpec,
    },
    Relu(CvValueId),
    Sigmoid(CvValueId),
    UpsampleNearest2d {
        src: CvValueId,
        spec: Upsample2dSpec,
    },
    Concat {
        inputs: Vec<CvValueId>,
        dim: i64,
    },
    YoloDecode {
        src: CvValueId,
        spec: YoloDecodeSpec,
    },
    Nms {
        src: CvValueId,
        spec: NmsSpec,
    },
}

#[derive(Debug)]
struct CvNode {
    op: CvOp,
}

#[derive(Debug, Default)]
pub struct CvProgram {
    nodes: Vec<CvNode>,
    output: Option<CvValueId>,
}

impl CvProgram {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(&mut self) -> CvValueId {
        self.push(CvOp::Input)
    }

    pub fn conv2d(&mut self, src: CvValueId, spec: Conv2dSpec) -> CvValueId {
        self.push(CvOp::Conv2d { src, spec })
    }

    pub fn relu(&mut self, src: CvValueId) -> CvValueId {
        self.push(CvOp::Relu(src))
    }

    pub fn sigmoid(&mut self, src: CvValueId) -> CvValueId {
        self.push(CvOp::Sigmoid(src))
    }

    pub fn upsample_nearest2d(&mut self, src: CvValueId, spec: Upsample2dSpec) -> CvValueId {
        self.push(CvOp::UpsampleNearest2d { src, spec })
    }

    pub fn concat(&mut self, inputs: &[CvValueId], dim: i64) -> CvValueId {
        self.push(CvOp::Concat {
            inputs: inputs.to_vec(),
            dim,
        })
    }

    pub fn yolo_decode(&mut self, src: CvValueId, spec: YoloDecodeSpec) -> CvValueId {
        self.push(CvOp::YoloDecode { src, spec })
    }

    pub fn nms(&mut self, src: CvValueId, spec: NmsSpec) -> CvValueId {
        self.push(CvOp::Nms { src, spec })
    }

    pub fn set_output(&mut self, id: CvValueId) {
        self.output = Some(id);
    }

    pub fn run(&self, input: &Tensor) -> Result<Tensor> {
        let out = self
            .output
            .ok_or_else(|| anyhow!("cv program has no output"))?;
        let mut values: Vec<Option<Tensor>> = (0..self.nodes.len()).map(|_| None).collect();

        for (idx, node) in self.nodes.iter().enumerate() {
            let v = match &node.op {
                CvOp::Input => input.shallow_clone(),
                CvOp::Conv2d { src, spec } => {
                    let x = get_value(&values, *src)?;
                    x.f_conv2d(
                        &spec.weight,
                        spec.bias.as_ref(),
                        spec.stride,
                        spec.padding,
                        spec.dilation,
                        spec.groups,
                    )?
                }
                CvOp::Relu(src) => {
                    let x = get_value(&values, *src)?;
                    x.relu()
                }
                CvOp::Sigmoid(src) => {
                    let x = get_value(&values, *src)?;
                    x.sigmoid()
                }
                CvOp::UpsampleNearest2d { src, spec } => {
                    let x = get_value(&values, *src)?;
                    let sz = x.size();
                    if sz.len() != 4 {
                        return Err(anyhow!(
                            "upsample_nearest2d expects NCHW tensor, got {:?}",
                            sz
                        ));
                    }
                    let out_h = sz[2] * spec.scale_h;
                    let out_w = sz[3] * spec.scale_w;
                    x.f_upsample_nearest2d([out_h, out_w], None, None)?
                }
                CvOp::Concat { inputs, dim } => {
                    if inputs.is_empty() {
                        return Err(anyhow!("concat requires at least one input"));
                    }
                    let tensors: Vec<Tensor> = inputs
                        .iter()
                        .map(|id| get_value(&values, *id).map(Tensor::shallow_clone))
                        .collect::<Result<Vec<_>>>()?;
                    Tensor::cat(&tensors, *dim)
                }
                CvOp::YoloDecode { src, spec } => {
                    let pred = get_value(&values, *src)?;
                    yolo_decode(pred, *spec)?
                }
                CvOp::Nms { src, spec } => {
                    let dets = get_value(&values, *src)?;
                    nms_cpu(dets, *spec)?
                }
            };
            values[idx] = Some(v);
        }

        get_value(&values, out).map(Tensor::shallow_clone)
    }

    fn push(&mut self, op: CvOp) -> CvValueId {
        let id = CvValueId(self.nodes.len());
        self.nodes.push(CvNode { op });
        id
    }
}

fn get_value(values: &[Option<Tensor>], id: CvValueId) -> Result<&Tensor> {
    values
        .get(id.index())
        .and_then(|x| x.as_ref())
        .ok_or_else(|| anyhow!("invalid cv node reference {}", id.index()))
}

pub fn yolo_decode(pred: &Tensor, spec: YoloDecodeSpec) -> Result<Tensor> {
    let sz = pred.size();
    if sz.is_empty() {
        return Err(anyhow!("yolo decode expects rank >= 2 tensor"));
    }
    let last = *sz.last().unwrap();
    if last < 6 {
        return Err(anyhow!(
            "yolo decode expects last dimension >= 6, got {}",
            last
        ));
    }

    // Flatten to [num_boxes, C] where C >= 6.
    let flat = pred.f_view([-1, last])?;

    let coords = flat.narrow(1, 0, 4);
    let x = coords.narrow(1, 0, 1);
    let y = coords.narrow(1, 1, 1);
    let w = coords.narrow(1, 2, 1);
    let h = coords.narrow(1, 3, 1);

    let (x1, y1, x2, y2) = if spec.input_is_xywh {
        (
            &x - (&w * 0.5),
            &y - (&h * 0.5),
            &x + (&w * 0.5),
            &y + (&h * 0.5),
        )
    } else {
        (x, y, w, h)
    };

    let obj = flat.narrow(1, 4, 1).sigmoid();
    let cls = flat.narrow(1, 5, last - 5).sigmoid();
    let (cls_conf, cls_idx) = cls.max_dim(1, true);
    let score = obj * cls_conf;
    let class_id = cls_idx.to_kind(Kind::Float);

    let out = Tensor::cat(&[x1, y1, x2, y2, score, class_id], 1);

    let keep = out.narrow(1, 4, 1).squeeze_dim(1).ge(spec.score_threshold);
    let idx = keep.nonzero();
    if idx.size()[0] == 0 {
        return Ok(Tensor::zeros([0, 6], (Kind::Float, pred.device())));
    }
    let idx = idx.squeeze_dim(1);
    Ok(out.index_select(0, &idx))
}

pub fn nms_cpu(dets: &Tensor, spec: NmsSpec) -> Result<Tensor> {
    let device = dets.device();
    let dets = dets
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous();
    let sz = dets.size();
    if sz.len() != 2 || sz[1] < 6 {
        return Err(anyhow!(
            "nms expects [N, >=6] decoded detections, got {:?}",
            sz
        ));
    }

    let rows = sz[0] as usize;
    let cols = sz[1] as usize;
    let flat: Vec<f32> = Vec::<f32>::try_from(dets.f_view([-1])?)?;

    let mut boxes = Vec::new();
    for i in 0..rows {
        let r = &flat[(i * cols)..((i + 1) * cols)];
        let score = r[4] as f64;
        if score < spec.score_threshold {
            continue;
        }
        boxes.push([r[0], r[1], r[2], r[3], r[4], r[5]]);
    }

    boxes.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep: Vec<[f32; 6]> = Vec::new();
    'outer: for b in boxes {
        if keep.len() >= spec.max_detections {
            break;
        }
        for k in &keep {
            if spec.class_aware && (b[5] - k[5]).abs() > f32::EPSILON {
                continue;
            }
            let iou = iou_xyxy(&b, k);
            if iou > spec.iou_threshold as f32 {
                continue 'outer;
            }
        }
        keep.push(b);
    }

    if keep.is_empty() {
        return Ok(Tensor::zeros([0, 6], (Kind::Float, device)));
    }

    let mut out = Vec::with_capacity(keep.len() * 6);
    for b in keep {
        out.extend_from_slice(&b);
    }

    Ok(Tensor::f_from_slice(&out)?
        .f_view([-1, 6])?
        .to_device(device)
        .to_kind(Kind::Float))
}

pub fn iou_xyxy(a: &[f32; 6], b: &[f32; 6]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;

    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

pub struct CvBuilder {
    program: CvProgram,
    input: Option<Tensor>,
    current: Option<CvValueId>,
}

impl CvBuilder {
    pub fn new() -> Self {
        Self {
            program: CvProgram::new(),
            input: None,
            current: None,
        }
    }

    pub fn input(mut self, input: Tensor) -> Self {
        let id = self.program.input();
        self.input = Some(input);
        self.current = Some(id);
        self
    }

    pub fn conv2d(mut self, weight: Tensor, cfg: Conv2dCfg) -> Result<Self> {
        let src = self
            .current
            .ok_or_else(|| anyhow!("conv2d requires a prior input/op"))?;
        let id = self.program.conv2d(
            src,
            Conv2dSpec {
                weight,
                bias: cfg.bias,
                stride: cfg.stride,
                padding: cfg.padding,
                dilation: cfg.dilation,
                groups: cfg.groups,
            },
        );
        self.current = Some(id);
        Ok(self)
    }

    pub fn relu(mut self) -> Result<Self> {
        let src = self
            .current
            .ok_or_else(|| anyhow!("relu requires a prior input/op"))?;
        let id = self.program.relu(src);
        self.current = Some(id);
        Ok(self)
    }

    pub fn sigmoid(mut self) -> Result<Self> {
        let src = self
            .current
            .ok_or_else(|| anyhow!("sigmoid requires a prior input/op"))?;
        let id = self.program.sigmoid(src);
        self.current = Some(id);
        Ok(self)
    }

    pub fn upsample2x(mut self) -> Result<Self> {
        let src = self
            .current
            .ok_or_else(|| anyhow!("upsample requires a prior input/op"))?;
        let id = self.program.upsample_nearest2d(
            src,
            Upsample2dSpec {
                scale_h: 2,
                scale_w: 2,
            },
        );
        self.current = Some(id);
        Ok(self)
    }

    pub fn yolo_decode(mut self, score_threshold: f64) -> Result<Self> {
        let src = self
            .current
            .ok_or_else(|| anyhow!("yolo_decode requires a prior input/op"))?;
        let id = self.program.yolo_decode(
            src,
            YoloDecodeSpec {
                input_is_xywh: true,
                score_threshold,
            },
        );
        self.current = Some(id);
        Ok(self)
    }

    pub fn nms(
        mut self,
        iou_threshold: f64,
        score_threshold: f64,
        max_detections: usize,
    ) -> Result<Self> {
        let src = self
            .current
            .ok_or_else(|| anyhow!("nms requires a prior input/op"))?;
        let id = self.program.nms(
            src,
            NmsSpec {
                iou_threshold,
                score_threshold,
                max_detections,
                class_aware: true,
            },
        );
        self.current = Some(id);
        Ok(self)
    }

    pub fn run(mut self) -> Result<Tensor> {
        let out = self
            .current
            .ok_or_else(|| anyhow!("run requires at least one op"))?;
        let input = self
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("run requires an input tensor"))?;
        self.program.set_output(out);
        self.program.run(input)
    }
}
