#include "DescriptorSet.h"


DescriptorSet::DescriptorSet(std::shared_ptr<VulkanContext> ctx,
                             std::shared_ptr<DescriptorSetLayout> layout)
    : _ctx(std::move(ctx)), _layout(std::move(layout))
{
    VkDescriptorSetLayout rawLayout = _layout->get();

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = _ctx->descriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &rawLayout;

    if (vkAllocateDescriptorSets(_ctx->device, &ai, &_ds) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }
}


DescriptorSet::~DescriptorSet()
{
    if (_ds != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(_ctx->device, _ctx->descriptorPool, 1, &_ds);
    }
}


DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : _ctx(std::move(other._ctx)),
      _layout(std::move(other._layout)),
      _ds(other._ds)
{
    other._ds = VK_NULL_HANDLE;
}


DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept
{
    if (this != &other) {
        if (_ds != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(_ctx->device, _ctx->descriptorPool, 1, &_ds);
        }
        _ctx = std::move(other._ctx);
        _layout = std::move(other._layout);
        _ds = other._ds;
        other._ds = VK_NULL_HANDLE;
    }
    return *this;
}


void DescriptorSet::update(const std::vector<DescriptorWrite>& writes)
{
    std::vector<VkWriteDescriptorSet> vkWrites;
    vkWrites.reserve(writes.size());

    for (const auto& w : writes) {
        VkWriteDescriptorSet vw{};
        vw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vw.dstSet = _ds;
        vw.dstBinding = w.binding;
        vw.dstArrayElement = w.arrayElement;
        vw.descriptorCount = 1;
        vw.descriptorType = _layout->typeOf(w.binding);

        if (std::holds_alternative<VkDescriptorBufferInfo>(w.info)) {
            vw.pBufferInfo = &std::get<VkDescriptorBufferInfo>(w.info);
        } else {
            vw.pImageInfo  = &std::get<VkDescriptorImageInfo>(w.info);
        }

        vkWrites.push_back(vw);
    }

    vkUpdateDescriptorSets(_ctx->device,
                           static_cast<uint32_t>(vkWrites.size()),
                           vkWrites.data(),
                           0, nullptr);
}
