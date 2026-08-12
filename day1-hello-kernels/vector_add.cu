#include <stdio.h>
#include <cuda_runtime.h>

// __global__ marks this as a "kernel" — a function that runs on the GPU,
// but is launched (called) from the CPU (host).
// Every GPU thread that gets launched runs this SAME function independently.
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    /* Each thread needs to figure out "which element of the array am I responsible for?"
    GPU threads are organized in a 2-level hierarchy: grid -> blocks -> threads.
      blockIdx.x  = which block this thread belongs to (0, 1, 2, ...)
      blockDim.x  = how many threads are in each block (we set this at launch)
      threadIdx.x = this thread's index within its own block (0 to blockDim.x-1)
    Multiplying blockIdx.x * blockDim.x gives the starting index of this block's
    chunk of the array, then + threadIdx.x picks this thread's specific element. */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard check: we may have launched more threads than array elements
    // (because we round up to a multiple of block size — see blocksPerGrid below).
    // Without this check, extra threads would read/write out of bounds.
    if (i < n) {
        c[i] = a[i] + b[i];  // the actual "work" — one thread does exactly one addition
    }
}

int main() {
    const int n = 1 << 20; // 1,048,576 elements (~1M). "1 << 20" = 2^20, a common way to write it.
    size_t bytes = n * sizeof(float);

    // --- HOST memory (regular RAM, CPU-accessible) ---
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Fill inputs with known values so we can check correctness later.
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // --- DEVICE memory (GPU's own RAM, the CPU cannot directly read/write this) ---
    // "d_" prefix is just a naming convention for "device pointer".
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);  // allocate space on the GPU
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy input data from host RAM -> device RAM.
    // This is a real, often-expensive step: the GPU can't see your CPU's memory directly,
    // everything it operates on has to be explicitly transferred over PCIe first.
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // --- Launch configuration ---
    int threadsPerBlock = 256;  // common choice; GPUs like block sizes that are multiples of 32 (warp size)
    // We need enough blocks to cover all n elements. Integer division rounds down,
    // so we add (threadsPerBlock - 1) before dividing to round UP instead —
    // this guarantees we launch at least n threads total, possibly a few extra
    // (which is why the "if (i < n)" guard inside the kernel matters).
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // The <<<...>>> syntax is CUDA-specific: it launches the kernel with the given
    // grid/block configuration. This call returns immediately — the CPU doesn't wait
    // for the GPU to finish (kernel launches are asynchronous).
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Since the launch is async, we explicitly wait here for the GPU to finish
    // before we try to read back the results. Without this, we might read stale/garbage data.
    cudaDeviceSynchronize();

    // Copy the result back from device -> host so the CPU can read it.
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sanity check: every element should be 1.0 + 2.0 = 3.0
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) { ok = false; break; }
    }
    printf(ok ? "CUDA vector add: PASSED (c[0]=%.1f)\n" : "CUDA vector add: FAILED\n", h_c[0]);

    // Always free what you allocate — GPU memory leaks are easy to cause and easy to forget.
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}