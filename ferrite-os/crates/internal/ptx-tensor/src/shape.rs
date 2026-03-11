//! Shape and stride types for tensors.

use smallvec::SmallVec;
use std::fmt;

/// Maximum number of dimensions stored inline (8 dims before heap allocation).
pub const MAX_INLINE_DIMS: usize = 8;

/// Shape type using small vector optimization.
pub type Shape = SmallVec<[usize; MAX_INLINE_DIMS]>;

/// Strides type using small vector optimization.
pub type Strides = SmallVec<[usize; MAX_INLINE_DIMS]>;

/// Calculate strides for a contiguous (row-major) tensor.
pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides = Strides::with_capacity(shape.len());
    if shape.is_empty() {
        return strides;
    }

    let mut stride: usize = 1;
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides.reverse();
    strides
}

/// Calculate the total number of elements from a shape (unchecked, may overflow).
pub fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Calculate the total number of elements with overflow checking.
/// Returns Err on overflow instead of silently wrapping.
pub fn checked_elem_count(shape: &[usize]) -> std::result::Result<usize, &'static str> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or("shape element count overflow")
    })
}

/// Check if strides indicate a contiguous tensor.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    if shape.is_empty() {
        return true;
    }

    let expected = contiguous_strides(shape);
    strides == expected.as_slice()
}

/// Compute the size in bytes for a tensor with given shape and element size.
pub fn size_bytes(shape: &[usize], elem_size: usize) -> usize {
    elem_count(shape) * elem_size
}

/// Compute the size in bytes with overflow checking.
pub fn checked_size_bytes(shape: &[usize], elem_size: usize) -> std::result::Result<usize, &'static str> {
    checked_elem_count(shape)?
        .checked_mul(elem_size)
        .ok_or("shape size_bytes overflow")
}

/// Broadcast shapes following NumPy broadcasting rules.
///
/// Returns the broadcast shape if compatible, None otherwise.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Shape> {
    let max_ndim = a.len().max(b.len());
    let mut result = Shape::with_capacity(max_ndim);

    // Iterate from the end (rightmost dimension)
    for i in 0..max_ndim {
        let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if dim_a == dim_b {
            result.push(dim_a);
        } else if dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            // Incompatible shapes
            return None;
        }
    }

    result.reverse();
    Some(result)
}

/// Check if two shapes are broadcastable.
pub fn can_broadcast(a: &[usize], b: &[usize]) -> bool {
    broadcast_shapes(a, b).is_some()
}

/// Calculate the reduction output shape.
pub fn reduce_shape(shape: &[usize], dim: usize, keepdim: bool) -> Shape {
    let mut result = Shape::from_slice(shape);
    if keepdim {
        result[dim] = 1;
    } else {
        result.remove(dim);
    }
    result
}

/// Calculate outer, reduce, and inner sizes for reduction operations.
pub fn reduction_sizes(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let reduce = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, reduce, inner)
}

/// Shape information for display.
#[derive(Debug, Clone)]
pub struct ShapeInfo {
    pub shape: Shape,
    pub strides: Strides,
    pub elem_count: usize,
    pub is_contiguous: bool,
}

impl ShapeInfo {
    pub fn new(shape: Shape, strides: Strides) -> Self {
        let elem_count = elem_count(&shape);
        let is_contiguous = is_contiguous(&shape, &strides);
        Self {
            shape,
            strides,
            elem_count,
            is_contiguous,
        }
    }

    pub fn contiguous(shape: Shape) -> Self {
        let strides = contiguous_strides(&shape);
        Self::new(shape, strides)
    }
}

impl fmt::Display for ShapeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        assert_eq!(contiguous_strides(&[2, 3, 4]).as_slice(), &[12, 4, 1]);
        assert_eq!(contiguous_strides(&[4]).as_slice(), &[1]);
        assert!(contiguous_strides(&[]).is_empty());
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(
            broadcast_shapes(&[2, 3], &[3]).unwrap().as_slice(),
            &[2, 3]
        );
        assert_eq!(
            broadcast_shapes(&[2, 1, 4], &[3, 4]).unwrap().as_slice(),
            &[2, 3, 4]
        );
        assert!(broadcast_shapes(&[2, 3], &[4]).is_none());
    }

    #[test]
    fn test_reduction_sizes() {
        let (outer, reduce, inner) = reduction_sizes(&[2, 3, 4], 1);
        assert_eq!(outer, 2);
        assert_eq!(reduce, 3);
        assert_eq!(inner, 4);
    }
}
