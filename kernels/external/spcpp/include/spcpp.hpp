#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <memory>
#include <functional>
#include <mutex>
#include <typeinfo>
#include <dlfcn.h>
#include <sys/stat.h>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>

namespace spc {

// ============================================================
// STATE REGISTRY - Persists across hot-reloads
// ============================================================
// State is stored in the main program's memory, not in modules.
// When a module reloads, it reconnects to the same state.

class StateRegistry {
public:
    static StateRegistry& get() {
        static StateRegistry instance;
        return instance;
    }

    // Get or create CPU state
    template<typename T>
    T* get_or_create(const std::string& name, T default_val) {
        std::lock_guard<std::mutex> lock(mtx_);

        auto it = cpu_state_.find(name);
        if (it == cpu_state_.end()) {
            // First time - allocate and initialize
            auto storage = std::make_unique<StateStorage>();
            storage->data.resize(sizeof(T));
            new (storage->data.data()) T(default_val);
            storage->size = sizeof(T);
            storage->type_name = typeid(T).name();

            T* ptr = reinterpret_cast<T*>(storage->data.data());
            cpu_state_[name] = std::move(storage);

            std::cout << "\033[1;35m[STATE]\033[0m Created: " << name << std::endl;
            return ptr;
        }

        // Existing state - return pointer
        return reinterpret_cast<T*>(it->second->data.data());
    }

    // Get or create GPU state (device pointer)
    template<typename T>
    T* get_or_create_gpu(const std::string& name, std::function<void(T**)> init_fn) {
        std::lock_guard<std::mutex> lock(mtx_);

        auto it = gpu_state_.find(name);
        if (it == gpu_state_.end()) {
            // First time - call initializer
            T* dev_ptr = nullptr;
            init_fn(&dev_ptr);
            gpu_state_[name] = reinterpret_cast<void*>(dev_ptr);

            std::cout << "\033[1;32m[GPU_STATE]\033[0m Allocated: " << name << std::endl;
            return dev_ptr;
        }

        // Existing - return same device pointer
        return reinterpret_cast<T*>(it->second);
    }

    // Check if state exists
    bool has(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx_);
        return cpu_state_.count(name) || gpu_state_.count(name);
    }

    // Clear all state (for testing)
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        cpu_state_.clear();
        // Note: GPU memory is NOT freed - user must manage that
        gpu_state_.clear();
    }

    // List all state (debugging)
    void list() {
        std::lock_guard<std::mutex> lock(mtx_);
        std::cout << "\033[1;36m[STATE REGISTRY]\033[0m" << std::endl;
        std::cout << "  CPU state:" << std::endl;
        for (auto& [name, storage] : cpu_state_) {
            std::cout << "    " << name << " (" << storage->size << " bytes)" << std::endl;
        }
        std::cout << "  GPU state:" << std::endl;
        for (auto& [name, ptr] : gpu_state_) {
            std::cout << "    " << name << " (device ptr: " << ptr << ")" << std::endl;
        }
    }

private:
    StateRegistry() = default;

    struct StateStorage {
        std::vector<char> data;
        size_t size;
        std::string type_name;
    };

    std::mutex mtx_;
    std::map<std::string, std::unique_ptr<StateStorage>> cpu_state_;
    std::map<std::string, void*> gpu_state_;
};

// ============================================================
// STATE MACROS
// ============================================================

// CPU state that persists across reloads
// Usage: STATE(float, learning_rate, 0.001f);
#define STATE(type, name, default_val) \
    static type& name = *spc::StateRegistry::get().get_or_create<type>(#name, default_val)

// GPU state (device pointer) that persists across reloads
// Usage: GPU_STATE(float*, d_weights, 1024 * sizeof(float));
// The size parameter is used for cudaMalloc
#define GPU_STATE(type, name, size_bytes) \
    static type& name = *spc::StateRegistry::get().get_or_create_gpu<type>( \
        #name, \
        [](type** ptr) { cudaMalloc(ptr, size_bytes); } \
    )

// GPU state with custom initializer
// Usage: GPU_STATE_INIT(float*, d_weights, [](float** p) { cudaMalloc(p, 1024); cudaMemset(*p, 0, 1024); });
#define GPU_STATE_INIT(type, name, init_fn) \
    static type& name = *spc::StateRegistry::get().get_or_create_gpu<type>(#name, init_fn)

// Array state for fixed-size arrays
// Usage: STATE_ARRAY(float, weights, 1024, 0.0f);
#define STATE_ARRAY(type, name, count, default_val) \
    static type* name = spc::StateRegistry::get().get_or_create<std::array<type, count>>( \
        #name, \
        []() { std::array<type, count> arr; arr.fill(default_val); return arr; }() \
    )->data()

namespace fs = std::filesystem;

#ifndef SPC_CUDA_PATH
#define SPC_CUDA_PATH "/usr/local/cuda"
#endif

#ifndef SPC_SM_VER
#define SPC_SM_VER "86"
#endif

#ifndef SPC_BUILD_DIR
#define SPC_BUILD_DIR "./build"
#endif

#ifndef SPC_SPCPP_DIR
#define SPC_SPCPP_DIR ""
#endif

// --- Build Directory Resolution ---
// Returns project-relative build directory based on source file location
inline std::string resolve_build_dir(const std::string& source_path) {
    fs::path src = fs::absolute(source_path);
    fs::path build_dir = src.parent_path() / "build";

    // Create directories if needed
    fs::create_directories(build_dir / "bin");
    fs::create_directories(build_dir / "lib");
    fs::create_directories(build_dir / "ptx");
    fs::create_directories(build_dir / "deps");

    return build_dir.string();
}

// Forward declarations
class Module;

class ModuleManager {
public:
    static ModuleManager& get() {
        static ModuleManager instance;
        return instance;
    }

    std::shared_ptr<Module> get_module(const std::string& path);
    std::shared_ptr<Module> reload_module(const std::string& path);
    bool needs_reload(const std::string& path);

private:
    std::map<std::string, std::shared_ptr<Module>> cache;
    std::map<std::string, std::filesystem::file_time_type> timestamps;
    std::mutex mtx_;
};

class Module {
public:
    Module(void* handle, const std::string& path) : handle(handle), path(path) {}
    ~Module() { if (handle) dlclose(handle); }

    template<typename T>
    T* get(const std::string& name) {
        void* sym = dlsym(handle, name.c_str());
        return reinterpret_cast<T*>(sym);
    }

    template<typename Ret, typename... Args>
    Ret call(const std::string& name, Args... args) {
        auto func = get<Ret(Args...)>(name);
        if (!func) throw std::runtime_error("Symbol not found: " + name + " in " + path);
        return func(args...);
    }

    struct Object {
        void* ptr;
        Module* mod;
        std::string name;

        template<typename Ret, typename... Args>
        Ret call(const std::string& method, Args... args) {
            return mod->call<Ret, void*, Args...>(name + "_" + method, ptr, args...);
        }

        void del() { mod->call<void, void*>(name + "_delete", ptr); }
    };

    Object create(const std::string& class_name, const char* arg = "") {
        void* p = call<void*, const char*>(class_name + "_new", arg);
        return {p, this, class_name};
    }

private:
    void* handle;
    std::string path;
};

inline std::shared_ptr<Module> ModuleManager::get_module(const std::string& source_path) {
    std::lock_guard<std::mutex> lock(mtx_);

    fs::path abs_source = fs::absolute(source_path);
    std::string abs_path = abs_source.string();

    if (cache.count(abs_path)) return cache[abs_path];

    // Build dir is relative to the source file's location
    std::string build_dir = resolve_build_dir(abs_path);
    std::string lib_name = abs_source.stem().string() + ".so";
    std::string lib_path = build_dir + "/lib/" + lib_name;
    std::string dep_path = build_dir + "/deps/" + abs_source.stem().string() + ".d";

    // Check if rebuild needed (source newer than lib, or deps changed)
    bool needs_rebuild = !fs::exists(lib_path) ||
                         fs::last_write_time(abs_source) > fs::last_write_time(lib_path);

    if (!needs_rebuild && fs::exists(dep_path)) {
        // Check header dependencies
        std::ifstream depfile(dep_path);
        std::string line;
        while (std::getline(depfile, line)) {
            // Parse .d file format - skip first line (target), check deps
            size_t pos = line.find(':');
            if (pos != std::string::npos) line = line.substr(pos + 1);

            std::istringstream iss(line);
            std::string dep;
            while (iss >> dep) {
                if (dep == "\\") continue;
                if (fs::exists(dep) && fs::last_write_time(dep) > fs::last_write_time(lib_path)) {
                    needs_rebuild = true;
                    break;
                }
            }
            if (needs_rebuild) break;
        }
    }

    if (needs_rebuild) {
        std::cout << "\033[1;34m[spcpp]\033[0m " << source_path << " -> " << lib_path << std::endl;

        // Include the source's directory, spcpp include dir, and CUDA
        std::string src_dir = abs_source.parent_path().string();
        std::string spcpp_inc = std::string(SPC_SPCPP_DIR).empty() ? "" : " -I" + std::string(SPC_SPCPP_DIR) + "/include";
        std::string cuda_inc = " -I" + std::string(SPC_CUDA_PATH) + "/include";

        std::string cmd = "g++ -MMD -MF " + dep_path +
                          " -shared -fPIC -O3 -std=c++17" +
                          " -I" + src_dir + spcpp_inc + cuda_inc +
                          " " + abs_path + " -o " + lib_path;
        int res = system(cmd.c_str());
        if (res != 0) throw std::runtime_error("Compile error: " + source_path);
    }

    void* h = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!h) throw std::runtime_error(std::string(dlerror()));

    auto mod = std::make_shared<Module>(h, abs_path);
    cache[abs_path] = mod;
    timestamps[abs_path] = fs::last_write_time(abs_source);
    return mod;
}

// Check if module source has changed since last load
inline bool ModuleManager::needs_reload(const std::string& source_path) {
    std::lock_guard<std::mutex> lock(mtx_);

    fs::path abs_source = fs::absolute(source_path);
    std::string abs_path = abs_source.string();

    if (!cache.count(abs_path)) return false;
    if (!timestamps.count(abs_path)) return true;

    return fs::last_write_time(abs_source) > timestamps[abs_path];
}

// Hot-reload a module (preserves STATE, reloads code)
inline std::shared_ptr<Module> ModuleManager::reload_module(const std::string& source_path) {
    std::lock_guard<std::mutex> lock(mtx_);

    fs::path abs_source = fs::absolute(source_path);
    std::string abs_path = abs_source.string();

    // Remove from cache (this will dlclose when refcount hits 0)
    cache.erase(abs_path);
    timestamps.erase(abs_path);

    // Unlock and re-acquire through get_module
    mtx_.unlock();
    auto mod = get_module(source_path);
    mtx_.lock();

    std::cout << "\033[1;33m[HOT-RELOAD]\033[0m " << source_path << std::endl;
    return mod;
}

// Global Import function
inline std::shared_ptr<Module> import_module(const std::string& path) {
    return ModuleManager::get().get_module(path);
}

// --- CUDA Support ---
class CudaModule {
public:
    CudaModule(CUmodule mod) : mod(mod) {}

    // 1D launch
    void launch(const std::string& name, int grid, int block, void** args, int shared_mem = 0) {
        CUfunction func;
        cuModuleGetFunction(&func, mod, name.c_str());
        cuLaunchKernel(func, grid, 1, 1, block, 1, 1, shared_mem, 0, args, 0);
        cudaDeviceSynchronize();
    }

    // 3D launch for more complex kernels
    void launch3d(const std::string& name,
                  int gx, int gy, int gz,
                  int bx, int by, int bz,
                  void** args, int shared_mem = 0) {
        CUfunction func;
        cuModuleGetFunction(&func, mod, name.c_str());
        cuLaunchKernel(func, gx, gy, gz, bx, by, bz, shared_mem, 0, args, 0);
        cudaDeviceSynchronize();
    }

    // 2D launch (common for matrix ops)
    void launch2d(const std::string& name,
                  int gx, int gy, int bx, int by,
                  void** args, int shared_mem = 0) {
        launch3d(name, gx, gy, 1, bx, by, 1, args, shared_mem);
    }

private:
    CUmodule mod;
};

inline CudaModule import_cuda(const std::string& source_path) {
    fs::path abs_source = fs::absolute(source_path);
    std::string abs_path = abs_source.string();

    // Build dir is relative to the source file's location
    std::string build_dir = resolve_build_dir(abs_path);
    std::string ptx_name = abs_source.stem().string() + ".ptx";
    std::string ptx_path = build_dir + "/ptx/" + ptx_name;

    if (!fs::exists(ptx_path) || fs::last_write_time(abs_source) > fs::last_write_time(ptx_path)) {
        std::cout << "\033[1;32m[spcpp-gpu]\033[0m " << source_path << " -> " << ptx_path << " (sm_" << SPC_SM_VER << ")" << std::endl;

        // Include source directory
        std::string src_dir = abs_source.parent_path().string();

        std::string cmd = std::string(SPC_CUDA_PATH) + "/bin/nvcc -ptx -arch=sm_" + SPC_SM_VER +
                          " -I" + src_dir + " " + abs_path + " -o " + ptx_path;
        int res = system(cmd.c_str());
        if (res != 0) throw std::runtime_error("NVCC Error: " + source_path);

        // Fix PTX version compatibility
        std::string fix_cmd = "sed -i 's/.version 8.[3-9]/.version 8.2/g' " + ptx_path;
        int dummy = system(fix_cmd.c_str());
        (void)dummy;
    }

    static bool init = false;
    if (!init) {
        cuInit(0);
        CUdevice dev;
        CUcontext ctx;
        cuDeviceGet(&dev, 0);
        cuDevicePrimaryCtxRetain(&ctx, dev);
        cuCtxSetCurrent(ctx);
        init = true;
    }

    CUmodule m;
    if (cuModuleLoad(&m, ptx_path.c_str()) != CUDA_SUCCESS) throw std::runtime_error("PTX Load fail: " + ptx_path);
    return CudaModule(m);
}

} // namespace spc

// ============================================================
// IMPORT MACROS
// ============================================================

#define IMPORT(path) spc::import_module(path)
#define IMPORT_GPU(path) spc::import_cuda(path)
#define EXPORT extern "C"

// Hot-reload: automatically reloads if source changed
#define IMPORT_HOT(path) spc::import_hot(path)

// Manual reload control
#define RELOAD(path) spc::ModuleManager::get().reload_module(path)
#define NEEDS_RELOAD(path) spc::ModuleManager::get().needs_reload(path)

// State inspection
#define STATE_LIST() spc::StateRegistry::get().list()
#define STATE_CLEAR() spc::StateRegistry::get().clear()

namespace spc {

// Import with automatic hot-reload detection
inline std::shared_ptr<Module> import_hot(const std::string& path) {
    if (ModuleManager::get().needs_reload(path)) {
        return ModuleManager::get().reload_module(path);
    }
    return ModuleManager::get().get_module(path);
}

} // namespace spc
