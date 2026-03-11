// ============================================================================
// Shared Memory / Persistent Segments Implementation
// ============================================================================

void* gpu_hot_shm_alloc(GPUHotRuntime* runtime, const char* name, size_t size) {
    if (!runtime || !name || !runtime->global_registry) return NULL;

    // Check if it already exists globally
    void* existing = gpu_hot_shm_open(runtime, name);
    if (existing) return existing;

    // Find free slot in global registry
    int slot = -1;
    for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
        if (!runtime->global_registry[i].active) {
            slot = i;
            break;
        }
    }

    if (slot == -1) return NULL;

    // Allocate device memory for IPC-exportable segment.
    // cudaMalloc (not cudaMallocManaged) is required for cudaIpcGetMemHandle.
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Device memory allocation for IPC failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    // Initialize to zero
    cudaMemset(ptr, 0, size);

    // Obtain a proper CUDA IPC handle for cross-process sharing.
    // This handle can be opened by other processes via cudaIpcOpenMemHandle.
    cudaIpcMemHandle_t ipc_handle;
    memset(&ipc_handle, 0, sizeof(ipc_handle));
    err = cudaIpcGetMemHandle(&ipc_handle, ptr);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        cudaFree(ptr);
        return NULL;
    }

    // Register globally with owner PID for audit traceability
    strncpy(runtime->global_registry[slot].name, name, GPU_HOT_MAX_NAME_LEN - 1);
    runtime->global_registry[slot].ipc_handle = ipc_handle;
    runtime->global_registry[slot].size = size;
    runtime->global_registry[slot].active = true;
    runtime->global_registry[slot].created_at = GetTickCount64();
    runtime->global_registry[slot].owner_pid = (uint32_t)getpid();
    runtime->global_registry[slot].ref_count = 1;
    runtime->shm_count++;

    printf("[Ferrite-OS] Created IPC memory segment: '%s' (%zu bytes) at %p [pid=%u]\n",
           name, size, ptr, (unsigned)getpid());
    return ptr;
}

void* gpu_hot_shm_open(GPUHotRuntime* runtime, const char* name) {
    if (!runtime || !name || !runtime->global_registry) return NULL;

    for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
        if (runtime->global_registry[i].active && strcmp(runtime->global_registry[i].name, name) == 0) {
            GPURegistryEntry* entry = &runtime->global_registry[i];

            // If the current process is the owner, the original device pointer
            // is valid in this address space — retrieve it via IPC handle.
            // For cross-process access, use cudaIpcOpenMemHandle.
            if (entry->owner_pid == (uint32_t)getpid()) {
                // Same process: open via IPC handle to get the mapped pointer
                void* ptr = NULL;
                cudaError_t err = cudaIpcOpenMemHandle(&ptr, entry->ipc_handle,
                                                        cudaIpcMemLazyEnablePeerAccess);
                if (err == cudaErrorMapBufferObjectFailed) {
                    // Already mapped in this process context — the original
                    // allocation pointer is still valid. Re-derive it from
                    // a fresh IPC open on a peer, or return the alloc pointer.
                    // For same-process, the alloc pointer is what we handed out.
                    // Fall through to IPC open below.
                }
                if (err == cudaSuccess && ptr) {
                    entry->ref_count++;
                    printf("[Ferrite-OS] Opened IPC segment (same-pid): '%s' at %p\n", name, ptr);
                    return ptr;
                }
            }

            // Cross-process: open the IPC handle to map it into this address space
            void* ptr = NULL;
            cudaError_t err = cudaIpcOpenMemHandle(&ptr, entry->ipc_handle,
                                                    cudaIpcMemLazyEnablePeerAccess);
            if (err == cudaSuccess && ptr) {
                entry->ref_count++;
                printf("[Ferrite-OS] Opened IPC segment (cross-pid): '%s' at %p [owner=%u]\n",
                       name, ptr, entry->owner_pid);
                return ptr;
            }

            printf("[Ferrite-OS] Error: cudaIpcOpenMemHandle failed for '%s': %s\n",
                   name, cudaGetErrorString(err));
            return NULL;
        }
    }
    return NULL;
}

void gpu_hot_shm_close(GPUHotRuntime* runtime, void* ptr) {
    if (ptr) cudaIpcCloseMemHandle(ptr);
}

void gpu_hot_shm_unlink(GPUHotRuntime* runtime, const char* name) {
    if (!runtime || !name || !runtime->global_registry) return;
    
    for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
        if (runtime->global_registry[i].active && strcmp(runtime->global_registry[i].name, name) == 0) {
            runtime->global_registry[i].active = false;
            runtime->shm_count--;
            printf("[Ferrite-OS] Unlinked global segment: '%s'\n", name);
            return;
        }
    }
}

bool gpu_hot_get_registry_entry(GPUHotRuntime* runtime, int index, char* name_out, size_t* size_out, bool* active_out, unsigned long long* created_out) {
    if (!runtime || !runtime->global_registry) return false;
    if (index < 0 || index >= GPU_HOT_MAX_NAMED_SEGMENTS) return false;
    
    GPURegistryEntry* entry = &runtime->global_registry[index];
    
    if (name_out) strncpy(name_out, entry->name, 63);
    if (size_out) *size_out = entry->size;
    if (active_out) *active_out = entry->active;
    if (created_out) *created_out = (unsigned long long)entry->created_at;
    
    return true;
}

// Priority-Aware Scheduling Implementation
cudaStream_t gpu_hot_get_priority_stream(GPUHotRuntime* runtime, int priority) {
    if (!runtime) return 0;
    
    if (priority == PTX_PRIORITY_REALTIME) {
        return runtime->streams[0]; // First stream is always high priority
    } else if (priority == PTX_PRIORITY_LOW) {
        return runtime->streams[runtime->num_streams - 1]; // Last stream
    } else {
        return runtime->streams[2]; // Normal priority starts at index 2
    }
}

// Get the system state pointer (accessible by GPU via Zero-Copy)
PTXSystemState* gpu_hot_get_system_state(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return NULL;
    
    void* d_ptr;
    cudaHostGetDevicePointer(&d_ptr, runtime->system_state, 0);
    return (PTXSystemState*)d_ptr;
}

// Get the system state pointer (accessible by Host)
PTXSystemState* gpu_hot_get_host_system_state(GPUHotRuntime* runtime) {
    if (!runtime) return NULL;
    return runtime->system_state;
}

void gpu_hot_reset_system_state(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return;

    PTXSystemState* state = runtime->system_state;
    uint32_t auth_token = state->auth_token;
    int active_processes = state->active_processes;

    memset(state, 0, sizeof(PTXSystemState));
    state->auth_token = auth_token;
    state->active_processes = active_processes;
}

void gpu_hot_flush_task_queue(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return;

    PTXSystemState* state = runtime->system_state;
    state->active_tasks = 0;
    state->max_priority_active = 0;
    memset(&state->queue, 0, sizeof(PTXTaskQueue));
}

void gpu_hot_reset_watchdog(GPUHotRuntime* runtime) {
    if (!runtime) return;
    runtime->watchdog_tripped = false;
    runtime->last_launch_time = 0;
    if (runtime->system_state) {
        runtime->system_state->watchdog_alert = false;
    }
}

void gpu_hot_clear_signal_mask(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return;
    runtime->system_state->signal_mask = 0;
}

void gpu_hot_get_system_snapshot(GPUHotRuntime* runtime, GPUHotSystemSnapshot* snapshot) {
    if (!runtime || !snapshot) return;
    if (!runtime->system_state) {
        memset(snapshot, 0, sizeof(GPUHotSystemSnapshot));
        return;
    }

    PTXSystemState* state = runtime->system_state;
    snapshot->total_ops = state->total_ops;
    snapshot->active_processes = state->active_processes;
    snapshot->active_tasks = state->active_tasks;
    snapshot->max_priority_active = state->max_priority_active;
    snapshot->total_vram_used = state->total_vram_used;
    snapshot->watchdog_alert = state->watchdog_alert;
    snapshot->kernel_running = state->kernel_running;
    snapshot->shutdown_requested = state->shutdown_requested;
    snapshot->active_priority_level = state->active_priority_level;
    snapshot->signal_mask = state->signal_mask;
    snapshot->interrupt_cnt = state->interrupt_cnt;
    snapshot->queue_head = state->queue.head;
    snapshot->queue_tail = state->queue.tail;
    snapshot->queue_lock = state->queue.lock;
}

// ============================================================================
// Context Export API
// ============================================================================

// Get the captured CUcontext (as void* for C-compatible headers)
void* gpu_hot_get_context(GPUHotRuntime* runtime) {
    if (!runtime) return NULL;
    return (void*)runtime->cu_context;
}

// Export the CUcontext pointer as an environment variable for external consumers
void gpu_hot_export_context(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->cu_context) return;

    char buf[32];
    snprintf(buf, sizeof(buf), "%p", (void*)runtime->cu_context);
    setenv("PTX_CONTEXT_PTR", buf, 1);

    printf("[Ferrite-OS] Context exported: PTX_CONTEXT_PTR=%s\n", buf);
}

// Forward declaration of the actual kernel boot logic
// Defined in kernels/os_kernel.cu
extern "C" void ptx_os_launch_persistent_kernel(GPUHotRuntime* runtime, PTXSystemState* d_state);
extern "C" void ptx_os_launch_persistent_kernel_orin_um(GPUHotRuntime* runtime, PTXSystemState* d_state);

void ptx_os_boot_persistent_kernel(GPUHotRuntime* runtime) {
    if (!runtime) return;

    PTXSystemState* d_state = gpu_hot_get_system_state(runtime);
    if (!d_state) {
        printf("[PTX-OS] [ERROR] Could not resolve GPU-side system state!\n");
        return;
    }

    if (runtime->use_orin_um_kernel) {
        ptx_os_launch_persistent_kernel_orin_um(runtime, d_state);
    } else {
        ptx_os_launch_persistent_kernel(runtime, d_state);
    }
}

// ============================================================================
// Per-Owner Allocation API
// ============================================================================

void* gpu_hot_alloc_owned(GPUHotRuntime* runtime, size_t size, uint32_t owner_id) {
    if (!runtime || !runtime->tlsf_allocator) return NULL;

    gpu_hot_poll_deferred_internal(runtime, 64);

    ptx_mutex_lock(&runtime->async_lock);
    void* ptr = ptx_tlsf_alloc_owned(runtime->tlsf_allocator, size, owner_id);
    ptx_mutex_unlock(&runtime->async_lock);
    return ptr;
}

void gpu_hot_free_owner(GPUHotRuntime* runtime, uint32_t owner_id) {
    if (!runtime || !runtime->tlsf_allocator) return;

    ptx_mutex_lock(&runtime->async_lock);
    ptx_tlsf_free_owner(runtime->tlsf_allocator, owner_id);
    ptx_mutex_unlock(&runtime->async_lock);
}

void gpu_hot_get_owner_stats(GPUHotRuntime* runtime, TLSFOwnerStats* stats) {
    if (!runtime || !runtime->tlsf_allocator || !stats) return;

    ptx_mutex_lock(&runtime->async_lock);
    ptx_tlsf_get_owner_stats(runtime->tlsf_allocator, stats);
    ptx_mutex_unlock(&runtime->async_lock);
}

// ============================================================================
// VMM Bridge
// ============================================================================

void gpu_hot_set_vmm(GPUHotRuntime* runtime, VMMState* vmm) {
    if (runtime) runtime->vmm = vmm;
}

// ============================================================================
// Allocation Event Log
// ============================================================================

void gpu_hot_get_alloc_events(GPUHotRuntime* runtime, TLSFEventRing* ring_out) {
    if (!runtime || !runtime->tlsf_allocator || !ring_out) return;

    ptx_mutex_lock(&runtime->async_lock);
    ptx_tlsf_get_events(runtime->tlsf_allocator, ring_out);
    ptx_mutex_unlock(&runtime->async_lock);
}
