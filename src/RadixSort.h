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

    // Records all Vulkan commands (4 passes x histogram+scan+scatter) into cmd.
    // inputBuffer must have VK_BUFFER_USAGE_STORAGE_BUFFER_BIT and hold at
    // least numElements × sizeof(uint32_t) bytes. After the submitted commands
    // complete on the GPU, inputBuffer contains the sorted data.
    void recordSort(VkCommandBuffer cmd, VkBuffer inputBuffer, uint32_t numElements);

private:
    std::shared_ptr<VulkanContext> _ctx;
    uint32_t _maxElements;

    std::unique_ptr<Buffer> _scratchBuf;  // device-local ping-pong buffer
    std::unique_ptr<Buffer> _histBuf;     // numBlocks*256 uint32, STORAGE+TRANSFER_DST

    std::shared_ptr<DescriptorSetLayout> _histLayout;
    std::shared_ptr<DescriptorSetLayout> _scanLayout;
    std::shared_ptr<DescriptorSetLayout> _scatterLayout;

    // _histDS[0]: src=inputBuffer, hist=_histBuf   (passes 0,2)
    // _histDS[1]: src=_scratchBuf, hist=_histBuf   (passes 1,3)
    std::unique_ptr<DescriptorSet> _histDS[2];

    // Scan descriptor set — binding is always _histBuf, wired once.
    std::unique_ptr<DescriptorSet> _scanDS;

    // _scatterDS[0]: src=inputBuffer, dst=_scratchBuf, hist=_histBuf  (passes 0,2)
    // _scatterDS[1]: src=_scratchBuf, dst=inputBuffer, hist=_histBuf  (passes 1,3)
    std::unique_ptr<DescriptorSet> _scatterDS[2];

    void createDescriptors();
    void updateDescriptorSets(VkBuffer inputBuffer);


    std::unique_ptr<ComputePipeline> _histPipeline;
    std::unique_ptr<ComputePipeline> _scanPipeline;
    std::unique_ptr<ComputePipeline> _scatterPipeline;
    void createPipelines();


    void recordPass(VkCommandBuffer cmd, uint32_t pass,
                    uint32_t numBlocks, uint32_t numElements,
                    uint32_t dispatchX, uint32_t dispatchY);
};
