#include "pch.h"
#include "utils/AssetPath.h"
#include "bench/RadixSorterCPU.h"
#include "bench/RadixSorterOpenMP.h"
#include "RadixSort.h"
#include "vulkan/VulkanHelper.h"
#include <iomanip>
#include <omp.h>


namespace fs = std::filesystem;

struct TestFile {
    fs::path path;
    std::string dtype; // "u32" or "f32" (parsed from filename, may be "?")
};


static std::vector<TestFile> discoverTestFiles(const fs::path& dir) {
    std::vector<TestFile> out;
    if (!fs::exists(dir)) return out;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".bin") continue;

        // Filename convention: data_<dist>_<dtype>_<count>.bin
        std::string name = entry.path().stem().string();
        std::string dtype = "?";
        if (name.find("_u32_") != std::string::npos) dtype = "u32";
        else if (name.find("_f32_") != std::string::npos) dtype = "f32";

        out.push_back({entry.path(), dtype});
    }
    // sort by filename
    std::sort(out.begin(), out.end(), [](const TestFile& a, const TestFile& b) {
        return a.path.filename() < b.path.filename();
    });
    return out;
}


static int promptIndex(const std::string& prompt, int maxExclusive) {
    while (true) {
        std::cout << prompt << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) return -1;
        try {
            int v = std::stoi(line);
            if (v >= 0 && v < maxExclusive) return v;
        } catch (...) {}
        std::cout << "  invalid, try again.\n";
    }
}


template <typename T>
static std::vector<T> loadBinary(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("failed to open: " + path.string());
    const std::streamsize bytes = f.tellg();
    if (bytes < 0 || bytes % sizeof(T) != 0) {
        throw std::runtime_error("file size not a multiple of element size: " + path.string());
    }
    f.seekg(0);
    std::vector<T> data(static_cast<size_t>(bytes) / sizeof(T));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}


// ---- Algorithms registry --------------------------------------------------
// Each entry: name, supported dtype, runner that returns elapsed nanoseconds.
struct Algorithm {
    std::string name;
    std::string dtype;
    std::function<int64_t(const fs::path&)> run;
};

static int64_t runRadixCPU_u32(const fs::path& path) {
    auto data = loadBinary<uint32_t>(path);
    spdlog::info("Loaded {} elements ({:.1f} MiB)", data.size(), data.size() * sizeof(uint32_t) / double(1 << 20));

    auto t0 = std::chrono::steady_clock::now();
    auto sorted = RadixSorterCPU::radixSort(data);
    auto t1 = std::chrono::steady_clock::now();

    if (!std::is_sorted(sorted.begin(), sorted.end())) {
        spdlog::error("Sort verification FAILED — output is not sorted!");
    } else {
        spdlog::info("Sort verified.");
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

static int64_t runRadixOpenMP_u32(const fs::path& path) {
    auto data = loadBinary<uint32_t>(path);
    spdlog::info("Loaded {} elements ({:.1f} MiB)", data.size(), data.size() * sizeof(uint32_t) / double(1 << 20));
    spdlog::info("OMP threads: max={}, in_parallel will use this", omp_get_max_threads());

    auto t0 = std::chrono::steady_clock::now();
    auto sorted = RadixSorterOpenMP::radixSort(data);
    auto t1 = std::chrono::steady_clock::now();

    if (!std::is_sorted(sorted.begin(), sorted.end())) {
        spdlog::error("Sort verification FAILED — output is not sorted!");
    } else {
        spdlog::info("Sort verified.");
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

static int64_t runRadixGPU_u32(const fs::path& path) {
    VulkanContextConfig cfg;
    cfg.appName            = "vkRadix-bench";
    cfg.requireComputeQueue = true;
    cfg.enableValidation   = true;
    auto ctx = std::make_shared<VulkanContext>(cfg);

    auto data = loadBinary<uint32_t>(path);
    const uint32_t count = static_cast<uint32_t>(data.size());
    const VkDeviceSize bytes = (VkDeviceSize)count * sizeof(uint32_t);
    spdlog::info("Loaded {} elements ({:.1f} MiB)", count, bytes / double(1 << 20));

    // Upload via staging buffer.
    Buffer stagingBuf(ctx, bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingBuf.copyData(data.data(), bytes);
    data.clear();

    Buffer deviceBuf(ctx, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VulkanHelper::copyBuffer(ctx, stagingBuf.getBuffer(), deviceBuf.getBuffer(), bytes);

    RadixSort sorter(ctx, count);

    // Allocate and record command buffer.
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx->device, &allocInfo, &cmd);

    // Begin and record the sort operation
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    sorter.recordSort(cmd, deviceBuf.getBuffer(), count);
    vkEndCommandBuffer(cmd);

    // Submit and time GPU execution.
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vkCreateFence(ctx->device, &fenceInfo, nullptr, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    auto t0 = std::chrono::steady_clock::now();
    vkQueueSubmit(ctx->computeQueue, 1, &submitInfo, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();

    vkDestroyFence(ctx->device, fence, nullptr);
    vkFreeCommandBuffers(ctx->device, ctx->commandPool, 1, &cmd);

    // Readback and verify.
    Buffer readbackBuf(ctx, bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanHelper::copyBuffer(ctx, deviceBuf.getBuffer(), readbackBuf.getBuffer(), bytes);

    const auto* result = reinterpret_cast<const uint32_t*>(readbackBuf.getMappedMemory());
    if (!std::is_sorted(result, result + count)) {
        spdlog::error("Sort verification FAILED — output is not sorted!");
    } else {
        spdlog::info("Sort verified.");
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}


static const std::vector<Algorithm>& algorithms() {
    static const std::vector<Algorithm> algos = {
        {"Radix Sort (CPU)", "u32", runRadixCPU_u32},
        {"Radix Sort (OpenMP)", "u32", runRadixOpenMP_u32},
        {"Radix Sort (GPU Vulkan Compute)", "u32", runRadixGPU_u32},
    };
    return algos;
}


int main(int argc, char* argv[]) {
    spdlog::level::level_enum log_level = spdlog::level::info;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-vvv") log_level = spdlog::level::trace;
        else if (arg == "-vv") log_level = spdlog::level::debug;
    }
    spdlog::set_level(log_level);

    std::cout << "vkRadix benchmark by @arminkz\n\n";

    // 1. Discover test files in <assets>/test/
    fs::path testDir = AssetPath::getInstance()->get("test");
    auto files = discoverTestFiles(testDir);
    if (files.empty()) {
        spdlog::error("No .bin files found in {}", testDir.string());
        spdlog::error("Generate some with: scripts/gen_data.py 10M");
        return EXIT_FAILURE;
    }

    std::cout << "Test files in " << testDir.string() << ":\n";
    for (size_t i = 0; i < files.size(); ++i) {
        auto sz = fs::file_size(files[i].path);
        std::cout << "  [" << i << "] " << files[i].path.filename().string()
                  << "  (dtype=" << files[i].dtype
                  << ", " << (sz / (1 << 20)) << " MiB)\n";
    }
    int fileIdx = promptIndex("Pick a file [0-" + std::to_string(files.size() - 1) + "]: ", (int)files.size());
    if (fileIdx < 0) return EXIT_FAILURE;
    const TestFile& chosenFile = files[fileIdx];

    // 2. Filter algorithms by dtype, then prompt.
    const auto& allAlgos = algorithms();
    std::vector<const Algorithm*> compatible;
    for (const auto& a : allAlgos) {
        if (chosenFile.dtype == "?" || a.dtype == chosenFile.dtype) compatible.push_back(&a);
    }
    if (compatible.empty()) {
        spdlog::error("No algorithm available for dtype '{}'", chosenFile.dtype);
        return EXIT_FAILURE;
    }

    std::cout << "\nAlgorithms compatible with dtype=" << chosenFile.dtype << ":\n";
    for (size_t i = 0; i < compatible.size(); ++i) {
        std::cout << "  [" << i << "] " << compatible[i]->name << "\n";
    }
    int algoIdx = promptIndex("Pick an algorithm [0-" + std::to_string(compatible.size() - 1) + "]: ", (int)compatible.size());
    if (algoIdx < 0) return EXIT_FAILURE;
    const Algorithm* chosenAlgo = compatible[algoIdx];

    // 3. Run + time.
    std::cout << "\nRunning " << chosenAlgo->name << " on " << chosenFile.path.filename().string() << "\n";
    int64_t ns;
    try {
        ns = chosenAlgo->run(chosenFile.path);
    } catch (const std::exception& e) {
        spdlog::error("Run failed: {}", e.what());
        return EXIT_FAILURE;
    }

    const double ms = ns / 1e6;
    const size_t bytes = fs::file_size(chosenFile.path);
    const size_t elems = bytes / 4; // both u32 and f32 are 4 bytes
    const double melems_per_sec = (elems / 1e6) / (ns / 1e9);

    std::cout << "\n--- Result ---\n";
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << ms << " ms\n";
    std::cout << "  Elements:    " << elems << "\n";
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(2) << melems_per_sec << " M elem/s\n";

    return EXIT_SUCCESS;
}
