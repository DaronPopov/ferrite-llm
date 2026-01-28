extern "C" __global__ void equal_f32(const float* a, const float* b, bool* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (a[i] == b[i]);
    }
}
