#include <cuda_runtime.h>
#include <stdio.h>

#include "gpu/gpu_hot_runtime.h"

#ifndef PTX_ENABLE_CDP
#define PTX_ENABLE_CDP 0
#endif

#ifdef PTX_KERNEL_QUIET
#define KLOG(...) ((void)0)
#else
#define KLOG(...) printf(__VA_ARGS__)
#endif

// Reuse the shared CDP system-state pointer defined in kernels/os_kernel.cu.
extern __device__ PTXSystemState* d_ptx_system_state;

// ============================================================================
// Orin Unified-Memory Persistent Kernel Branch
//
// This mirrors the baseline PTX scheduler logic while pacing fences/yields for
// integrated-memory devices (Jetson Orin class).
// ============================================================================

__device__ __forceinline__ void ptx_orin_scheduler_backoff() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(256);
#else
    // Fallback for older architectures in fatbin builds where __nanosleep is unavailable.
    uint64_t start = clock64();
    while ((clock64() - start) < 1024ULL) {
    }
#endif
}

__global__ void ptx_os_kernel_orin_um(PTXSystemState* state) {
    uint32_t tid = threadIdx.x;

    if (tid == 0) {
        state->kernel_running = true;
        state->shutdown_requested = false;
        d_ptx_system_state = state;
        printf("[GPU-OS-ORIN-UM] Kernel initialized. Unified-memory scheduler active.\n");
    }

    __syncthreads();

    uint64_t iterations = 0;
    while (!state->shutdown_requested) {
        __syncthreads();

        if (tid == 0) {
            if (state->queue.head != state->queue.tail) {
                int best_idx = -1;
                int min_priority = 256;

                uint32_t current_tail = state->queue.tail;
                uint32_t current_head = state->queue.head;

                for (uint32_t i = current_tail; i != current_head; i++) {
                    PTXOSTask* candidate = &state->queue.tasks[i % PTX_MAX_QUEUE_SIZE];
                    if (candidate->active && !candidate->completed) {
                        if (candidate->priority < min_priority) {
                            min_priority = candidate->priority;
                            best_idx = i % PTX_MAX_QUEUE_SIZE;
                            if (min_priority == 0) break;
                        }
                    }
                }

                if (best_idx != -1) {
                    PTXOSTask* task = &state->queue.tasks[best_idx];
                    state->active_priority_level = task->priority;

                    switch (task->opcode) {
                        case 0: // NOP
                            break;

                        case 1: { // COMPUTE
                            typedef void (*compute_fn_t)(void**, int);
                            compute_fn_t fn = (compute_fn_t)task->args[0];
                            if (fn) {
#if PTX_ENABLE_CDP
                                fn<<<1, 256>>>(&task->args[1], task->priority);
#else
                                KLOG("[PTX-CDP] COMPUTE dispatch skipped (PTX_ENABLE_CDP=0)\n");
#endif
                            }
                        } break;

                        case 3: // SHUTDOWN
                            state->shutdown_requested = true;
                            break;

                        case 4: // SWAP_IN
                            KLOG("[GPU-OS-ORIN-UM] SWAP_IN task_id=%d\n", task->task_id);
                            atomicOr((unsigned long long*)&state->signal_mask, 0x4ULL);
                            break;

                        case 5: // VFS_MOUNT
                            state->fs_node_count++;
                            KLOG("[GPU-OS-ORIN-UM] VFS mount idx=%d nodes=%d\n",
                                   (int)(size_t)task->args[0], state->fs_node_count);
                            break;

                        case 6: // INTERRUPT
                            atomicOr((unsigned long long*)&state->signal_mask,
                                     (unsigned long long)task->args[0]);
                            break;

                        case 7: { // LAUNCH_KERNEL / CDP recursion
                            PTXKernelLaunch* launch = (PTXKernelLaunch*)&task->args[0];
                            if (launch && launch->kernel_func) {
#if PTX_ENABLE_CDP
                                typedef void (*kernel_ptr_t)(void**);
                                kernel_ptr_t func = (kernel_ptr_t)launch->kernel_func;
                                func<<<launch->grid, launch->block, launch->shared_mem, launch->stream>>>(
                                    launch->arg_values);
#else
                                KLOG("[PTX-CDP] Recursive launch skipped (PTX_ENABLE_CDP=0)\n");
#endif
                            }
                        } break;

                        default:
                            KLOG("[GPU-OS-ORIN-UM] Unknown opcode: %d\n", task->opcode);
                            break;
                    }

                    task->completed = true;
                    task->active = false;
                    state->total_ops++;

                    if (best_idx == (current_tail % PTX_MAX_QUEUE_SIZE)) {
                        state->queue.tail++;
                    }
                }
            }

            if (state->signal_mask != 0) {
                uint64_t signals = atomicExch((unsigned long long*)&state->signal_mask, 0);
                if (signals & 0x1) {
                    KLOG("[GPU-OS-ORIN-UM] Heartbeat pulse\n");
                }
                if (signals & 0x2) {
                    printf("[GPU-OS-ORIN-UM] Emergency flush requested\n");
                }
                state->interrupt_cnt++;
            }
        }

        if (tid == 0 && iterations % 20000000ULL == 0ULL) {
            KLOG("[GPU-OS-ORIN-UM-DEBUG] Head=%d Tail=%d Ops=%lld\n",
                   state->queue.head, state->queue.tail, state->total_ops);
        }

        // Unified-memory pacing: fewer system-wide fences plus short nanosleeps.
        if ((iterations & 0x3FFULL) == 0ULL) {
            __threadfence_system();
        } else {
            __threadfence();
        }
        if (tid == 0) {
            ptx_orin_scheduler_backoff();
        }
        iterations++;
    }

    if (tid == 0) {
        state->kernel_running = false;
        printf("[GPU-OS-ORIN-UM] Kernel shutdown.\n");
    }
}

extern "C" void ptx_os_launch_persistent_kernel_orin_um(GPUHotRuntime* runtime,
                                                         PTXSystemState* d_state) {
    (void)runtime;
    int block_size = 256;
    int os_blocks = 1;

    printf("[PTX-OS-ORIN-UM] Launching persistent kernel (%d blocks x %d threads)\n",
           os_blocks, block_size);
    ptx_os_kernel_orin_um<<<os_blocks, block_size, 0, 0>>>(d_state);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[PTX-OS-ORIN-UM] [ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[PTX-OS-ORIN-UM] Persistent kernel resident.\n");
}
