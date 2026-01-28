// compute.cu - EDIT THIS FILE WHILE RUNNING!
//
// The transform() kernel runs every frame.
// Change it and save - see results instantly!
//
// Variables available:
//   data[i] - the value at index i (read/write)
//   n       - total number of elements (1024)
//   t       - time (increases each frame)
//   i       - current thread index

extern "C" __global__ void transform(float* data, int n, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = (float)i / n;  // normalized position [0, 1]

    // =====================================================
    // EDIT BELOW - Try different formulas!
    // =====================================================

    // Simple sine wave
    data[i] = sinf(x * 10.0f + t);

    // TRY THESE (uncomment one):
    //
    // Moving wave packet:
    // data[i] = sinf(x * 20.0f - t * 3.0f) * expf(-powf(x - 0.5f - 0.3f*sinf(t), 2.0f) * 20.0f);
    //
    // Interference pattern:
    // data[i] = sinf(x * 15.0f + t) + sinf(x * 17.0f - t * 1.3f);
    //
    // Sawtooth:
    // data[i] = fmodf(x * 5.0f + t, 1.0f);
    //
    // Noise-ish:
    // data[i] = sinf(x * 100.0f + t) * sinf(x * 77.0f - t);
    //
    // Pulse train:
    // data[i] = (fmodf(x * 8.0f + t, 1.0f) < 0.3f) ? 1.0f : -1.0f;
    //
    // Exponential decay:
    // data[i] = expf(-x * 3.0f) * sinf(t * 5.0f);
    //
    // Your own formula:
    // data[i] = ???

    // =====================================================
}

// Bonus: add more kernels and call them from main.cpp!
extern "C" __global__ void reset(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = (float)i / n;
}
