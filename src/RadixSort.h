#pragma once
#include "pch.h"
#include "vulkan/VulkanContext.h"
#include "vulkan/ComputePipeline.h"
#include "vulkan/DescriptorSet.h"
#include "vulkan/DescriptorSetLayout.h"
#include "vulkan/resources/Buffer.h"

// Push Constants
struct RadixSortPC {
    uint32_t pass;
    uint32_t numBlocks;
    uint32_t numElements;
};


class RadixSort {
public:
    // Allocates internal scratch and histogram buffers sized for maxElements.
    RadixSort(std::shared_ptr<VulkanContext> ctx, uint32_t maxElements);
    ~RadixSort() = default;

    RadixSort(const RadixSort&) = delete;
    RadixSort& operator=(const RadixSort&) = delete;

    // Key-only sort. inputBuffer is sorted in-place after submitted commands complete.
    void recordSort(VkCommandBuffer cmd, VkBuffer inputBuffer, uint32_t numElements);

    // Key-value sort. Keys are sorted; values are permuted identically.
    void recordSort(VkCommandBuffer cmd, VkBuffer keyBuf, VkBuffer valBuf, uint32_t numElements);

private:
    std::shared_ptr<VulkanContext> _ctx;
    uint32_t _maxElements;

    std::unique_ptr<Buffer> _scratchBuf;     // key ping-pong scratch
    std::unique_ptr<Buffer> _scratchValBuf;  // value ping-pong scratch (KV mode)
    std::unique_ptr<Buffer> _histBuf;        // numBlocks*256 uint32, STORAGE+TRANSFER_DST

    std::shared_ptr<DescriptorSetLayout> _histLayout;
    std::shared_ptr<DescriptorSetLayout> _scanLayout;
    std::shared_ptr<DescriptorSetLayout> _scatterLayout;
    std::shared_ptr<DescriptorSetLayout> _scatterKVLayout;

    // _histDS[0]: src=keyBuf,     hist=_histBuf   (passes 0,2)
    // _histDS[1]: src=_scratchBuf, hist=_histBuf  (passes 1,3)
    std::unique_ptr<DescriptorSet> _histDS[2];

    // Scan descriptor set — binding is always _histBuf, wired once.
    std::unique_ptr<DescriptorSet> _scanDS;

    // _scatterDS[0]: src=keyBuf,      dst=_scratchBuf, hist=_histBuf  (passes 0,2)
    // _scatterDS[1]: src=_scratchBuf, dst=keyBuf,      hist=_histBuf  (passes 1,3)
    std::unique_ptr<DescriptorSet> _scatterDS[2];

    // KV variants — same ping-pong pattern, additionally carry valBuf / _scratchValBuf
    std::unique_ptr<DescriptorSet> _scatterKVDS[2];

    void createDescriptors();
    void updateDescriptorSets(VkBuffer keyBuf);
    void updateKVDescriptorSets(VkBuffer keyBuf, VkBuffer valBuf);

    std::unique_ptr<ComputePipeline> _histPipeline;
    std::unique_ptr<ComputePipeline> _scanPipeline;
    std::unique_ptr<ComputePipeline> _scatterPipeline;
    std::unique_ptr<ComputePipeline> _scatterKVPipeline;
    void createPipelines();

    void recordPass(VkCommandBuffer cmd, uint32_t pass,
                    uint32_t numBlocks, uint32_t numElements,
                    uint32_t dispatchX, uint32_t dispatchY,
                    bool kv = false);
};
