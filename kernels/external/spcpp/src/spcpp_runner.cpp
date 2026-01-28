// spcpp_runner.cpp - Cross-platform build tool
// Compile this once, use everywhere
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <chrono>

namespace fs = std::filesystem;

// ============================================================
// PLATFORM DETECTION
// ============================================================
#ifdef _WIN32
    #define SPCPP_WINDOWS
    #define EXE_EXT ".exe"
    #define LIB_EXT ".dll"
    #define PATH_SEP "\\"
    #include <windows.h>
#else
    #define SPCPP_LINUX
    #define EXE_EXT ""
    #define LIB_EXT ".so"
    #define PATH_SEP "/"
#endif

using namespace std;

// ============================================================
// GPU DETECTION (forward declaration for Config::detect)
// ============================================================
string detect_sm_version() {
    // Try nvidia-smi first (most reliable)
    FILE* pipe = popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1", "r");
    if (pipe) {
        char buffer[128];
        string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);

        // Parse "8.6" -> "86"
        if (!result.empty()) {
            string sm;
            for (char c : result) {
                if (c >= '0' && c <= '9') sm += c;
            }
            if (sm.length() >= 2) {
                cout << "\033[1;32m[spcpp]\033[0m Detected GPU compute capability: "
                     << sm[0] << "." << sm.substr(1) << " (sm_" << sm << ")" << endl;
                return sm;
            }
        }
    }

    // Fallback: try common SM versions in order of likelihood
    cout << "\033[1;33m[spcpp]\033[0m Could not detect GPU, defaulting to sm_86 (Ampere)" << endl;
    return "86";
}

// ============================================================
// CONFIGURATION
// ============================================================
struct Config {
    string cpp_compiler;
    string nvcc;
    string cuda_include;
    string cuda_lib;
    string sm_version;
    string spcpp_include;
    vector<string> extra_includes;
    vector<string> extra_libs;

    static Config detect(const string& spcpp_dir) {
        Config cfg;
        cfg.spcpp_include = spcpp_dir + PATH_SEP "include";

#ifdef SPCPP_WINDOWS
        cfg.cpp_compiler = "cl.exe";

        // Try to find CUDA
        const char* cuda_env = getenv("CUDA_PATH");
        string cuda_path = cuda_env ? cuda_env : "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0";

        cfg.nvcc = cuda_path + "\\bin\\nvcc.exe";
        cfg.cuda_include = cuda_path + "\\include";
        cfg.cuda_lib = cuda_path + "\\lib\\x64";

        cfg.extra_libs = {"cuda.lib", "cudart.lib", "cublas.lib", "cublasLt.lib"};
#else
        cfg.cpp_compiler = "g++";

        // Find CUDA
        const char* cuda_env = getenv("CUDA_HOME");
        if (!cuda_env) cuda_env = getenv("CUDA_PATH");
        string cuda_path = cuda_env ? cuda_env : "/usr/local/cuda";

        cfg.nvcc = cuda_path + "/bin/nvcc";
        cfg.cuda_include = cuda_path + "/include";
        cfg.cuda_lib = cuda_path + "/lib64";

        cfg.extra_libs = {"-lcuda", "-lcudart", "-lcublas", "-lcublasLt", "-ldl"};
#endif

        // Auto-detect SM version from GPU
        cfg.sm_version = detect_sm_version();

        return cfg;
    }
};

// ============================================================
// UTILITY FUNCTIONS
// ============================================================
int run(const string& cmd) {
    cout << "\033[90m$ " << cmd << "\033[0m" << endl;
    return system(cmd.c_str());
}

bool needs_rebuild(const string& source, const string& target) {
    if (!fs::exists(target)) return true;
    return fs::last_write_time(source) > fs::last_write_time(target);
}

string get_build_dir(const string& source_path) {
    fs::path src = fs::absolute(source_path);
    fs::path build = src.parent_path() / "build";

    fs::create_directories(build / "bin");
    fs::create_directories(build / "lib");
    fs::create_directories(build / "ptx");
    fs::create_directories(build / "deps");

    return build.string();
}

// ============================================================
// BUILD COMMANDS
// ============================================================
int build_main(const Config& cfg, const string& source, const string& output) {
    ostringstream cmd;

#ifdef SPCPP_WINDOWS
    cmd << "\"" << cfg.cpp_compiler << "\" /O2 /EHsc /std:c++17"
        << " /I\"" << cfg.spcpp_include << "\""
        << " /I\"" << cfg.cuda_include << "\""
        << " \"" << source << "\""
        << " /Fe\"" << output << "\""
        << " /link /LIBPATH:\"" << cfg.cuda_lib << "\"";
    for (auto& lib : cfg.extra_libs) cmd << " " << lib;
#else
    cmd << cfg.cpp_compiler << " -O3 -std=c++17"
        << " -I" << cfg.spcpp_include
        << " -I" << cfg.cuda_include
        << " " << source
        << " -o " << output
        << " -L" << cfg.cuda_lib
        << " -Wl,-rpath," << cfg.cuda_lib;
    for (auto& lib : cfg.extra_libs) cmd << " " << lib;
#endif

    return run(cmd.str());
}

int build_module(const Config& cfg, const string& source, const string& output) {
    ostringstream cmd;
    fs::path src_dir = fs::path(source).parent_path();

#ifdef SPCPP_WINDOWS
    cmd << "\"" << cfg.cpp_compiler << "\" /LD /O2 /EHsc /std:c++17"
        << " /I\"" << cfg.spcpp_include << "\""
        << " /I\"" << cfg.cuda_include << "\""
        << " /I\"" << src_dir.string() << "\""
        << " \"" << source << "\""
        << " /Fe\"" << output << "\"";
#else
    cmd << cfg.cpp_compiler << " -shared -fPIC -O3 -std=c++17"
        << " -I" << cfg.spcpp_include
        << " -I" << cfg.cuda_include
        << " -I" << src_dir.string()
        << " " << source
        << " -o " << output;
#endif

    return run(cmd.str());
}

int build_cuda(const Config& cfg, const string& source, const string& output) {
    ostringstream cmd;
    fs::path src_dir = fs::path(source).parent_path();

#ifdef SPCPP_WINDOWS
    cmd << "\"" << cfg.nvcc << "\" -ptx"
        << " -arch=sm_" << cfg.sm_version
        << " -I\"" << src_dir.string() << "\""
        << " \"" << source << "\""
        << " -o \"" << output << "\"";
#else
    cmd << cfg.nvcc << " -ptx"
        << " -arch=sm_" << cfg.sm_version
        << " -I" << src_dir.string()
        << " " << source
        << " -o " << output;
#endif

    return run(cmd.str());
}

// ============================================================
// MAIN
// ============================================================
void print_usage() {
    cout << "spcpp - Spatial C++ Runner\n\n";
    cout << "Usage:\n";
    cout << "  spcpp <source.cpp> [args...]    Build and run\n";
    cout << "  spcpp build <source.cpp>        Build only\n";
    cout << "  spcpp clean [directory]         Remove build artifacts\n";
    cout << "  spcpp --help                    Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    string arg1 = argv[1];

    // Get spcpp directory (where this binary lives)
    fs::path exe_path = fs::absolute(argv[0]);
    string spcpp_dir = exe_path.parent_path().parent_path().string();

    Config cfg = Config::detect(spcpp_dir);

    // Handle commands
    if (arg1 == "--help" || arg1 == "-h") {
        print_usage();
        return 0;
    }

    if (arg1 == "clean") {
        string target = argc > 2 ? argv[2] : ".";
        fs::path build_dir = fs::absolute(target) / "build";
        if (fs::exists(build_dir)) {
            cout << "\033[1;33m[spcpp-clean]\033[0m Removing " << build_dir << endl;
            fs::remove_all(build_dir);
            cout << "\033[1;32m[spcpp-clean]\033[0m Done" << endl;
        } else {
            cout << "\033[1;33m[spcpp-clean]\033[0m No build directory at " << build_dir << endl;
        }
        return 0;
    }

    if (arg1 == "build") {
        if (argc < 3) {
            cerr << "Error: No source file specified" << endl;
            return 1;
        }
        arg1 = argv[2];
        // Fall through to build
    }

    // Build and run source file
    if (!fs::exists(arg1)) {
        cerr << "\033[1;31m[spcpp-error]\033[0m Source not found: " << arg1 << endl;
        return 1;
    }

    fs::path source = fs::absolute(arg1);
    string build_dir = get_build_dir(source.string());

    string binary_name = source.stem().string() + EXE_EXT;
    string binary = build_dir + PATH_SEP "bin" PATH_SEP + binary_name;

    if (needs_rebuild(source.string(), binary)) {
        cout << "\033[1;36m[spcpp-build]\033[0m " << source.string() << " -> " << binary << endl;

        int res = build_main(cfg, source.string(), binary);
        if (res != 0) {
            cerr << "\033[1;31m[spcpp-error]\033[0m Build failed" << endl;
            return res;
        }
    }

    // Run (if not build-only mode)
    if (string(argv[1]) != "build") {
        // Change to source directory for relative imports
        fs::current_path(source.parent_path());

        // Build command with args
        ostringstream cmd;
#ifdef SPCPP_WINDOWS
        cmd << "\"" << binary << "\"";
#else
        cmd << binary;
#endif
        for (int i = 2; i < argc; i++) {
            cmd << " " << argv[i];
        }

        return system(cmd.str().c_str());
    }

    return 0;
}
