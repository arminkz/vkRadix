#include "RadixSorterOpenMP.h"
#include <omp.h>


static constexpr int RADIX_BITS = 8;
static constexpr int RADIX = 1 << RADIX_BITS;   // 256
static constexpr int PASSES = 4;                // sizeof(uint32_t) * 8 / RADIX_BITS


static void countingSortPass(const std::vector<uint32_t>& in,
                             std::vector<uint32_t>& out,
                             int shift)
{
    const size_t n = in.size();
    const int P = omp_get_max_threads();

    // Per-thread local histograms: P x RADIX, row-major.
    std::vector<uint32_t> hist(static_cast<size_t>(P) * RADIX, 0);

    // Parallel counting step: each thread creates a local histogram
    #pragma omp parallel
    {
        const int t = omp_get_thread_num();

        // the number of elements each thread is responsible for
        const size_t chunk = (n + P - 1) / P;

        // range for this thread
        const size_t lo = std::min(static_cast<size_t>(t) * chunk, n);
        const size_t hi = std::min(lo + chunk, n);

        // range in hist table that this thread is resposible for
        uint32_t* h = hist.data() + static_cast<size_t>(t) * RADIX;

        // count the digits (histogram)
        for (size_t i = lo; i < hi; ++i) {
            ++h[(in[i] >> shift) & (RADIX - 1)];
        }
    }

    // Serial prefix scan: convert the P x RADIX matrix into per-thread,
    // per-bucket starting offsets in the output. Bucket-major scan order
    // ensures all of bucket b's outputs come before bucket b+1's.
    uint32_t running = 0;
    for (int b = 0; b < RADIX; ++b) {
        for (int t = 0; t < P; ++t) {
            uint32_t c = hist[static_cast<size_t>(t) * RADIX + b];
            hist[static_cast<size_t>(t) * RADIX + b] = running;
            running += c;
        }
    }

    // Parallel scatter into out using the per-thread offsets.
    #pragma omp parallel
    {
        const int t = omp_get_thread_num();

        // the number of elements each thread is responsible for
        const size_t chunk = (n + P - 1) / P;

        // range for this thread
        const size_t lo = std::min(static_cast<size_t>(t) * chunk, n);
        const size_t hi = std::min(lo + chunk, n);

        // range in hist table that this thread is resposible for
        uint32_t* h = hist.data() + static_cast<size_t>(t) * RADIX;

        // Placement
        for (size_t i = lo; i < hi; ++i) {
            uint32_t bucket = (in[i] >> shift) & (RADIX - 1);
            out[h[bucket]++] = in[i];
        }
    }
}


std::vector<uint32_t> RadixSorterOpenMP::radixSort(const std::vector<uint32_t>& arr) {
    if (arr.empty()) return {};

    std::vector<uint32_t> a = arr;
    std::vector<uint32_t> b(arr.size());
    for (int p = 0; p < PASSES; ++p) {
        countingSortPass(a, b, p * RADIX_BITS);
        std::swap(a, b); //ping-pong
    }
    return a;
}
