#include "RadixSort.h"
#include "utils/AssetPath.h"
#include "vulkan/VulkanHelper.h"

static constexpr uint32_t RADIX       = 256;   // 8bits
static constexpr uint32_t PASSES      = 4;     // 32bits (uint32) / 8bits (radix)
static constexpr uint32_t BLOCK_SIZE  = 16384; // 256 (number of threads per WG) * 64 (number of elements per thread)
static constexpr uint32_t DISPATCH_X  = 1024;


RadixSort::RadixSort(std::shared_ptr<VulkanContext> ctx, uint32_t maxElements)
    : _ctx(ctx), _maxElements(maxElements)
{
    // Determine number of workgroups
    // each workgroup (256 threads) will work on one block, each thread will handle 64 elements
    const uint32_t maxNumBlocks = (maxElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    _scratchBuf = std::make_unique<Buffer>(
        _ctx,
        (VkDeviceSize)maxElements * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    _histBuf = std::make_unique<Buffer>(
        _ctx,
        (VkDeviceSize)maxNumBlocks * RADIX * sizeof(uint32_t), // rows: blocks  cols: radix digits
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    createDescriptors();
    createPipelines();
}


void RadixSort::createDescriptors()
{
    // all descriptors are SSBOs, only binding number changes 
    auto ssbo = [](uint32_t b) {
        return DescriptorSetLayout::Binding{
            b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    };

    _histLayout = std::make_shared<DescriptorSetLayout>(_ctx,
                    std::vector<DescriptorSetLayout::Binding>{ssbo(0), ssbo(1)});
    _scanLayout = std::make_shared<DescriptorSetLayout>(_ctx,
                    std::vector<DescriptorSetLayout::Binding>{ssbo(0)});
    _scatterLayout = std::make_shared<DescriptorSetLayout>(_ctx,
                    std::vector<DescriptorSetLayout::Binding>{ssbo(0), ssbo(1), ssbo(2)});

    for (int i = 0; i < 2; ++i) {
        _histDS[i]    = std::make_unique<DescriptorSet>(_ctx, _histLayout);
        _scatterDS[i] = std::make_unique<DescriptorSet>(_ctx, _scatterLayout);
    }
    _scanDS = std::make_unique<DescriptorSet>(_ctx, _scanLayout);

    // Scan binding is fixed for the lifetime of this object.
    _scanDS->update({{0, _histBuf->getDescriptorInfo()}});
}


void RadixSort::createPipelines()
{
    VkPushConstantRange histScatterPC{};
    histScatterPC.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    histScatterPC.offset = 0;
    histScatterPC.size = sizeof(RadixSortPC);

    VkPushConstantRange scanPC{};
    scanPC.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    scanPC.offset = 0;
    scanPC.size = sizeof(uint32_t);

    _histPipeline = std::make_unique<ComputePipeline>(
        _ctx,
        AssetPath::getInstance()->get("spv/radix_histogram_comp.spv"),
        std::vector<VkDescriptorSetLayout>{_histLayout->get()},
        std::vector<VkPushConstantRange>{histScatterPC}
    );

    _scanPipeline = std::make_unique<ComputePipeline>(
        _ctx,
        AssetPath::getInstance()->get("spv/radix_scan_comp.spv"),
        std::vector<VkDescriptorSetLayout>{_scanLayout->get()},
        std::vector<VkPushConstantRange>{scanPC}
    );

    _scatterPipeline = std::make_unique<ComputePipeline>(
        _ctx,
        AssetPath::getInstance()->get("spv/radix_scatter_comp.spv"),
        std::vector<VkDescriptorSetLayout>{_scatterLayout->get()},
        std::vector<VkPushConstantRange>{histScatterPC}
    );
}


void RadixSort::updateDescriptorSets(VkBuffer inputBuffer)
{
    VkDescriptorBufferInfo inBuf  { inputBuffer,              0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo scrBuf { _scratchBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo hisBuf { _histBuf->getBuffer(),    0, VK_WHOLE_SIZE };

    _histDS[0]   ->update({{0, inBuf},  {1, hisBuf}});
    _histDS[1]   ->update({{0, scrBuf}, {1, hisBuf}});
    _scatterDS[0]->update({{0, inBuf},  {1, scrBuf}, {2, hisBuf}});
    _scatterDS[1]->update({{0, scrBuf}, {1, inBuf},  {2, hisBuf}});
}


void RadixSort::recordPass(VkCommandBuffer cmd, uint32_t pass,
                           uint32_t numBlocks, uint32_t numElements,
                           uint32_t dispatchX, uint32_t dispatchY)
{
    // Zero out the histogram buffer 
    vkCmdFillBuffer(cmd, _histBuf->getBuffer(), 0, VK_WHOLE_SIZE, 0u);
    VulkanHelper::barrierFillToCompute(cmd);

    RadixSortPC pc{pass, numBlocks, numElements};

    // Dispatch histogram pass
    _histPipeline->bind(cmd);
    VkDescriptorSet histDS = _histDS[pass & 1u]->get();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _histPipeline->getPipelineLayout(),
                            0, 1, &histDS, 0, nullptr);
    vkCmdPushConstants(cmd, _histPipeline->getPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, dispatchX, dispatchY, 1);

    VulkanHelper::barrierComputeToCompute(cmd);

    // Dispatch prefix scan (single workgroup, 256 threads)
    _scanPipeline->bind(cmd);
    VkDescriptorSet scanDS = _scanDS->get();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _scanPipeline->getPipelineLayout(),
                            0, 1, &scanDS, 0, nullptr);
    vkCmdPushConstants(cmd, _scanPipeline->getPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &numBlocks);
    vkCmdDispatch(cmd, 1, 1, 1);

    VulkanHelper::barrierComputeToCompute(cmd);

    // Dispatch scatter pass
    _scatterPipeline->bind(cmd);
    VkDescriptorSet scatterDS = _scatterDS[pass & 1u]->get();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _scatterPipeline->getPipelineLayout(),
                            0, 1, &scatterDS, 0, nullptr);
    vkCmdPushConstants(cmd, _scatterPipeline->getPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, dispatchX, dispatchY, 1);
}


void RadixSort::recordSort(VkCommandBuffer cmd, VkBuffer inputBuffer, uint32_t numElements)
{
    updateDescriptorSets(inputBuffer);

    const uint32_t numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint32_t dispatchX = std::min(numBlocks, DISPATCH_X);
    const uint32_t dispatchY = (numBlocks + DISPATCH_X - 1) / DISPATCH_X;

    for (uint32_t p = 0; p < PASSES; ++p) {
        recordPass(cmd, p, numBlocks, numElements, dispatchX, dispatchY);
        if (p != PASSES - 1) {
            VulkanHelper::barrierComputeToCompute(cmd);
        }
    }
}
