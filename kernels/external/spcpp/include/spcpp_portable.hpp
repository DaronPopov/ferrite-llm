// spcpp_portable.hpp - Cross-platform runtime (Windows + Linux)
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <mutex>
#include <array>
#include <typeinfo>
#include <filesystem>
#include <cstdlib>

// Platform detection
#ifdef _WIN32
    #define SPCPP_WINDOWS
    #include <windows.h>
    #define SPCPP_LIB_EXT ".dll"
    #define SPCPP_PATH_SEP "\\"
#else
    #define SPCPP_LINUX
    #include <dlfcn.h>
    #define SPCPP_LIB_EXT ".so"
    #define SPCPP_PATH_SEP "/"
#endif

// CUDA headers (same on both platforms)
#include <cuda.h>
#include <cuda_runtime.h>

namespace spc {

namespace fs = std::filesystem;

// ============================================================
// STATE REGISTRY - Persists across hot-reloads
// ============================================================

class StateRegistry {
public:
    static StateRegistry& get() {
        static StateRegistry instance;
        return instance;
    }

    template<typename T>
    T* get_or_create(const std::string& name, T default_val) {
        std::lock_guard<std::mutex> lock(mtx_);

        auto it = cpu_state_.find(name);
        if (it == cpu_state_.end()) {
            auto storage = std::make_unique<StateStorage>();
            storage->data.resize(sizeof(T));
            new (storage->data.data()) T(default_val);
            storage->size = sizeof(T);

            T* ptr = reinterpret_cast<T*>(storage->data.data());
            cpu_state_[name] = std::move(storage);

            std::cout << "\033[1;35m[STATE]\033[0m Created: " << name << std::endl;
            return ptr;
        }
        return reinterpret_cast<T*>(it->second->data.data());
    }

    template<typename T>
    T* get_or_create_gpu(const std::string& name, std::function<void(T**)> init_fn) {
        std::lock_guard<std::mutex> lock(mtx_);

        auto it = gpu_state_.find(name);
        if (it == gpu_state_.end()) {
            T* dev_ptr = nullptr;
            init_fn(&dev_ptr);
            gpu_state_[name] = reinterpret_cast<void*>(dev_ptr);

            std::cout << "\033[1;32m[GPU_STATE]\033[0m Allocated: " << name << std::endl;
            return dev_ptr;
        }
        return reinterpret_cast<T*>(it->second);
    }

    bool has(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx_);
        return cpu_state_.count(name) || gpu_state_.count(name);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        cpu_state_.clear();
        gpu_state_.clear();
    }

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
    };

    std::mutex mtx_;
    std::map<std::string, std::unique_ptr<StateStorage>> cpu_state_;
    std::map<std::string, void*> gpu_state_;
};

// ============================================================
// PLATFORM ABSTRACTION - Dynamic Library Loading
// ============================================================

class DynLib {
    void* handle_ = nullptr;
    std::string path_;

public:
    DynLib() = default;

    explicit DynLib(const std::string& path) : path_(path) {
#ifdef SPCPP_WINDOWS
        handle_ = LoadLibraryA(path.c_str());
        if (!handle_) {
            throw std::runtime_error("LoadLibrary failed: " + path + " (error " + std::to_string(GetLastError()) + ")");
        }
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle_) {
            throw std::runtime_error(std::string(dlerror()));
        }
#endif
    }

    ~DynLib() {
        if (handle_) {
#ifdef SPCPP_WINDOWS
            FreeLibrary((HMODULE)handle_);
#else
            dlclose(handle_);
#endif
        }
    }

    // Move only
    DynLib(DynLib&& other) noexcept : handle_(other.handle_), path_(std::move(other.path_)) {
        other.handle_ = nullptr;
    }
    DynLib& operator=(DynLib&& other) noexcept {
        if (this != &other) {
            if (handle_) {
#ifdef SPCPP_WINDOWS
                FreeLibrary((HMODULE)handle_);
#else
                dlclose(handle_);
#endif
            }
            handle_ = other.handle_;
            path_ = std::move(other.path_);
            other.handle_ = nullptr;
        }
        return *this;
    }
    DynLib(const DynLib&) = delete;
    DynLib& operator=(const DynLib&) = delete;

    void* get_symbol(const std::string& name) {
#ifdef SPCPP_WINDOWS
        return (void*)GetProcAddress((HMODULE)handle_, name.c_str());
#else
        return dlsym(handle_, name.c_str());
#endif
    }

    template<typename T>
    T* get(const std::string& name) {
        return reinterpret_cast<T*>(get_symbol(name));
    }
};

// ============================================================
// PLATFORM ABSTRACTION - Compilation
// ============================================================

struct CompilerConfig {
    std::string cpp_compiler;
    std::string nvcc_path;
    std::string cuda_include;
    std::string cuda_lib;
    std::string sm_version;

    static CompilerConfig detect() {
        CompilerConfig cfg;

#ifdef SPCPP_WINDOWS
        // Windows: Use MSVC cl.exe or g++ from MinGW
        cfg.cpp_compiler = "cl.exe";  // Or detect g++ from PATH

        // CUDA paths on Windows
        const char* cuda_path = std::getenv("CUDA_PATH");
        if (cuda_path) {
            cfg.nvcc_path = std::string(cuda_path) + "\\bin\\nvcc.exe";
            cfg.cuda_include = std::string(cuda_path) + "\\include";
            cfg.cuda_lib = std::string(cuda_path) + "\\lib\\x64";
        } else {
            cfg.nvcc_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\nvcc.exe";
            cfg.cuda_include = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\include";
            cfg.cuda_lib = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\lib\\x64";
        }
#else
        // Linux: Use g++
        cfg.cpp_compiler = "g++";

        // Find CUDA
        const char* cuda_path = std::getenv("CUDA_HOME");
        if (!cuda_path) cuda_path = std::getenv("CUDA_PATH");
        std::string cuda_base = cuda_path ? cuda_path : "/usr/local/cuda";

        cfg.nvcc_path = cuda_base + "/bin/nvcc";
        cfg.cuda_include = cuda_base + "/include";
        cfg.cuda_lib = cuda_base + "/lib64";
#endif

        // Auto-detect SM version from GPU
        cfg.sm_version = "86";  // Default
        FILE* pipe = popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1", "r");
        if (pipe) {
            char buffer[128];
            std::string result;
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
            pclose(pipe);
            // Parse "8.6" -> "86"
            std::string sm;
            for (char c : result) {
                if (c >= '0' && c <= '9') sm += c;
            }
            if (sm.length() >= 2) {
                cfg.sm_version = sm;
                std::cout << "\033[1;32m[spcpp]\033[0m Auto-detected GPU: sm_" << sm << std::endl;
            }
        }

        return cfg;
    }
};

inline int run_command(const std::string& cmd) {
#ifdef SPCPP_WINDOWS
    return system(cmd.c_str());
#else
    return system(cmd.c_str());
#endif
}

// ============================================================
// BUILD DIRECTORY MANAGEMENT
// ============================================================

inline std::string get_build_dir(const std::string& source_path) {
    fs::path src = fs::absolute(source_path);
    fs::path build_dir = src.parent_path() / "build";

    fs::create_directories(build_dir / "bin");
    fs::create_directories(build_dir / "lib");
    fs::create_directories(build_dir / "ptx");
    fs::create_directories(build_dir / "obj");

    return build_dir.string();
}

// ============================================================
// MODULE - Loaded shared library with symbol access
// ============================================================

class Module {
    DynLib lib_;
    std::string source_path_;

public:
    Module(DynLib&& lib, const std::string& path)
        : lib_(std::move(lib)), source_path_(path) {}

    template<typename T>
    T* get(const std::string& name) {
        return lib_.get<T>(name);
    }

    template<typename Ret, typename... Args>
    Ret call(const std::string& name, Args... args) {
        auto func = get<Ret(Args...)>(name);
        if (!func) {
            throw std::runtime_error("Symbol not found: " + name);
        }
        return func(args...);
    }

    // Object creation helper
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
};

// ============================================================
// MODULE MANAGER - Caches and compiles modules
// ============================================================

class ModuleManager {
    std::map<std::string, std::shared_ptr<Module>> cache_;
    std::map<std::string, fs::file_time_type> timestamps_;
    CompilerConfig config_;
    std::mutex mtx_;

public:
    static ModuleManager& get() {
        static ModuleManager instance;
        return instance;
    }

    ModuleManager() : config_(CompilerConfig::detect()) {}

    bool needs_reload(const std::string& source_path) {
        std::lock_guard<std::mutex> lock(mtx_);
        fs::path abs_source = fs::absolute(source_path);
        std::string abs_path = abs_source.string();

        if (!cache_.count(abs_path)) return false;
        if (!timestamps_.count(abs_path)) return true;
        return fs::last_write_time(abs_source) > timestamps_[abs_path];
    }

    std::shared_ptr<Module> reload(const std::string& source_path) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            fs::path abs_source = fs::absolute(source_path);
            std::string abs_path = abs_source.string();
            cache_.erase(abs_path);
            timestamps_.erase(abs_path);
        }
        std::cout << "\033[1;33m[HOT-RELOAD]\033[0m " << source_path << std::endl;
        return load(source_path);
    }

    std::shared_ptr<Module> load(const std::string& source_path) {
        std::lock_guard<std::mutex> lock(mtx_);
        fs::path abs_source = fs::absolute(source_path);
        std::string abs_path = abs_source.string();

        if (cache_.count(abs_path)) {
            return cache_[abs_path];
        }

        std::string build_dir = get_build_dir(abs_path);
        std::string lib_name = abs_source.stem().string() + SPCPP_LIB_EXT;
        std::string lib_path = build_dir + SPCPP_PATH_SEP "lib" SPCPP_PATH_SEP + lib_name;

        // Check if rebuild needed
        bool needs_rebuild = !fs::exists(lib_path) ||
                            fs::last_write_time(abs_source) > fs::last_write_time(lib_path);

        if (needs_rebuild) {
            std::cout << "\033[1;34m[spcpp]\033[0m " << source_path << " -> " << lib_path << std::endl;

            std::string src_dir = abs_source.parent_path().string();

#ifdef SPCPP_WINDOWS
            // Windows compilation with cl.exe
            std::string cmd = config_.cpp_compiler + " /LD /O2 /EHsc"
                " /I\"" + src_dir + "\""
                " /I\"" + config_.cuda_include + "\""
                " \"" + abs_path + "\""
                " /Fe\"" + lib_path + "\""
                " /link /LIBPATH:\"" + config_.cuda_lib + "\" cuda.lib cudart.lib";
#else
            // Linux compilation with g++
            std::string cmd = config_.cpp_compiler + " -shared -fPIC -O3 -std=c++17"
                " -I" + src_dir +
                " -I" + config_.cuda_include +
                " " + abs_path +
                " -o " + lib_path;
#endif

            int res = run_command(cmd);
            if (res != 0) {
                throw std::runtime_error("Compilation failed: " + source_path);
            }
        }

        auto mod = std::make_shared<Module>(DynLib(lib_path), abs_path);
        cache_[abs_path] = mod;
        timestamps_[abs_path] = fs::last_write_time(abs_source);
        return mod;
    }
};

// Import with hot-reload detection
inline std::shared_ptr<Module> import_hot(const std::string& path) {
    if (ModuleManager::get().needs_reload(path)) {
        return ModuleManager::get().reload(path);
    }
    return ModuleManager::get().load(path);
}

// ============================================================
// CUDA MODULE - PTX loading and kernel launch
// ============================================================

class CudaModule {
    CUmodule mod_;

public:
    CudaModule(CUmodule mod) : mod_(mod) {}

    void launch(const std::string& name, int grid, int block, void** args, int shared_mem = 0) {
        CUfunction func;
        CUresult res = cuModuleGetFunction(&func, mod_, name.c_str());
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error("Kernel not found: " + name);
        }
        cuLaunchKernel(func, grid, 1, 1, block, 1, 1, shared_mem, 0, args, 0);
        cudaDeviceSynchronize();
    }

    void launch2d(const std::string& name, int gx, int gy, int bx, int by,
                  void** args, int shared_mem = 0) {
        CUfunction func;
        cuModuleGetFunction(&func, mod_, name.c_str());
        cuLaunchKernel(func, gx, gy, 1, bx, by, 1, shared_mem, 0, args, 0);
        cudaDeviceSynchronize();
    }

    void launch3d(const std::string& name, int gx, int gy, int gz,
                  int bx, int by, int bz, void** args, int shared_mem = 0) {
        CUfunction func;
        cuModuleGetFunction(&func, mod_, name.c_str());
        cuLaunchKernel(func, gx, gy, gz, bx, by, bz, shared_mem, 0, args, 0);
        cudaDeviceSynchronize();
    }
};

// CUDA context initialization
inline void init_cuda() {
    static bool initialized = false;
    if (!initialized) {
        cuInit(0);
        CUdevice dev;
        CUcontext ctx;
        cuDeviceGet(&dev, 0);
        cuDevicePrimaryCtxRetain(&ctx, dev);
        cuCtxSetCurrent(ctx);
        initialized = true;
    }
}

// CUDA module loader
inline CudaModule load_cuda(const std::string& source_path) {
    init_cuda();

    fs::path abs_source = fs::absolute(source_path);
    std::string abs_path = abs_source.string();

    std::string build_dir = get_build_dir(abs_path);
    std::string ptx_name = abs_source.stem().string() + ".ptx";
    std::string ptx_path = build_dir + SPCPP_PATH_SEP "ptx" SPCPP_PATH_SEP + ptx_name;

    CompilerConfig config = CompilerConfig::detect();

    bool needs_rebuild = !fs::exists(ptx_path) ||
                        fs::last_write_time(abs_source) > fs::last_write_time(ptx_path);

    if (needs_rebuild) {
        std::cout << "\033[1;32m[spcpp-gpu]\033[0m " << source_path << " -> " << ptx_path
                  << " (sm_" << config.sm_version << ")" << std::endl;

        std::string src_dir = abs_source.parent_path().string();

#ifdef SPCPP_WINDOWS
        std::string cmd = "\"" + config.nvcc_path + "\" -ptx"
            " -arch=sm_" + config.sm_version +
            " -I\"" + src_dir + "\""
            " \"" + abs_path + "\""
            " -o \"" + ptx_path + "\"";
#else
        std::string cmd = config.nvcc_path + " -ptx"
            " -arch=sm_" + config.sm_version +
            " -I" + src_dir +
            " " + abs_path +
            " -o " + ptx_path;
#endif

        int res = run_command(cmd);
        if (res != 0) {
            throw std::runtime_error("NVCC failed: " + source_path);
        }

        // Fix PTX version compatibility (CUDA driver version issues)
#ifndef SPCPP_WINDOWS
        std::string fix_cmd = "sed -i 's/.version 8.[3-9]/.version 8.2/g' " + ptx_path;
        run_command(fix_cmd);
#endif
    }

    CUmodule mod;
    CUresult res = cuModuleLoad(&mod, ptx_path.c_str());
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("PTX load failed: " + ptx_path);
    }

    return CudaModule(mod);
}

// ============================================================
// CONVENIENCE MACROS
// ============================================================

inline std::shared_ptr<Module> import_module(const std::string& path) {
    return ModuleManager::get().load(path);
}

} // namespace spc

// ============================================================
// USER-FACING MACROS
// ============================================================

// Module imports
#define IMPORT(path) spc::import_module(path)
#define IMPORT_GPU(path) spc::load_cuda(path)
#define IMPORT_HOT(path) spc::import_hot(path)
#define EXPORT extern "C"

// Hot-reload control
#define RELOAD(path) spc::ModuleManager::get().reload(path)
#define NEEDS_RELOAD(path) spc::ModuleManager::get().needs_reload(path)

// CPU state that persists across reloads
#define STATE(type, name, default_val) \
    static type& name = *spc::StateRegistry::get().get_or_create<type>(#name, default_val)

// GPU state (device pointer) that persists across reloads
#define GPU_STATE(type, name, size_bytes) \
    static type& name = *spc::StateRegistry::get().get_or_create_gpu<type>( \
        #name, \
        [](type** ptr) { cudaMalloc(ptr, size_bytes); } \
    )

// GPU state with custom initializer
#define GPU_STATE_INIT(type, name, init_fn) \
    static type& name = *spc::StateRegistry::get().get_or_create_gpu<type>(#name, init_fn)

// State inspection
#define STATE_LIST() spc::StateRegistry::get().list()
#define STATE_CLEAR() spc::StateRegistry::get().clear()

// Windows DLL export macro
#ifdef SPCPP_WINDOWS
    #define SPCPP_API __declspec(dllexport)
#else
    #define SPCPP_API
#endif
