#pragma once
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

namespace spc {

/**
 * HotPool - Zero-Copy Shared VRAM for Inter-Process Communication (IPC)
 */
class HotPool {
public:
    struct Meta {
        cudaIpcMemHandle_t handle;
        size_t size;
        bool initialized;
    };

    HotPool(const std::string& name, size_t size_mb = 0, bool is_master = false) : name(name), is_master(is_master) {
        std::string shm_name = "/" + name + "_hotpool_shm";
        
        if (is_master) {
            shm_unlink(shm_name.c_str());
            shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
            int res = ftruncate(shm_fd, sizeof(Meta));
            (void)res;
        } else {
            shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
            if (shm_fd < 0) throw std::runtime_error("HotPool: Could not find master pool: " + name);
        }

        meta = (Meta*)mmap(NULL, sizeof(Meta), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

        if (is_master) {
            size_t bytes = size_mb * 1024 * 1024;
            cudaMalloc(&d_ptr, bytes);
            cudaIpcGetMemHandle(&meta->handle, d_ptr);
            meta->size = bytes;
            meta->initialized = true;
            std::cout << "\033[1;32m[HotPool]\033[0m Master Pool Created: " << name << " (" << size_mb << "MB)" << std::endl;
        } else {
            while(!meta->initialized) usleep(1000); // Wait for master
            cudaIpcOpenMemHandle(&d_ptr, meta->handle, cudaIpcMemLazyEnablePeerAccess);
            std::cout << "\033[1;36m[HotPool]\033[0m Attached to Pool: " << name << std::endl;
        }
    }

    void* data() { return d_ptr; }
    size_t size() const { return meta->size; }

    ~HotPool() {
        if (is_master) {
            cudaFree(d_ptr);
            shm_unlink(("/" + name + "_hotpool_shm").c_str());
        } else {
            cudaIpcCloseMemHandle(d_ptr);
        }
        munmap(meta, sizeof(Meta));
        close(shm_fd);
    }

private:
    std::string name;
    bool is_master;
    int shm_fd;
    void* d_ptr;
    Meta* meta;
};

} // namespace spc
