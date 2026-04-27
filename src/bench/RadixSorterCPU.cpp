#include "RadixSorterCPU.h"


static constexpr int RADIX_BITS = 8;
static constexpr int RADIX = 1 << RADIX_BITS;   // 256
static constexpr int PASSES = 4;                // sizeof(uint32_t) * 8 / RADIX_BITS

static void countingSort(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, int shift) {

    uint32_t count[RADIX] = {0};

    for (uint32_t v : in) {
        ++count[(v >> shift) & (RADIX - 1)];
    }

    for (int idx = 1; idx < RADIX; ++idx) {
        count[idx] += count[idx - 1];
    }

    for (int idx = static_cast<int>(in.size()) - 1; idx >= 0; --idx) {
        uint32_t bucket = (in[idx] >> shift) & (RADIX - 1);
        out[count[bucket] - 1] = in[idx];
        --count[bucket];
    }
}


std::vector<uint32_t> RadixSorterCPU::radixSort(const std::vector<uint32_t>& arr) {
    if (arr.empty()) return {};

    std::vector<uint32_t> a = arr;
    std::vector<uint32_t> b(arr.size());
    for (int p = 0; p < PASSES; ++p) {
        countingSort(a, b, p * RADIX_BITS);
        std::swap(a, b);
    }
    return a; //PASSES should be even
}
