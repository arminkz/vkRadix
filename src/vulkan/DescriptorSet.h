#pragma once
#include "pch.h"
#include "VulkanContext.h"
#include "DescriptorSetLayout.h"
#include <variant>


struct DescriptorWrite {
    uint32_t binding;
    std::variant<VkDescriptorBufferInfo, VkDescriptorImageInfo> info;
    uint32_t arrayElement = 0;
};


class DescriptorSet
{
public:
    DescriptorSet(std::shared_ptr<VulkanContext> ctx,
                  std::shared_ptr<DescriptorSetLayout> layout);
    ~DescriptorSet();

    DescriptorSet(const DescriptorSet&) = delete;
    DescriptorSet& operator=(const DescriptorSet&) = delete;

    DescriptorSet(DescriptorSet&& other) noexcept;
    DescriptorSet& operator=(DescriptorSet&& other) noexcept;

    // Issues vkUpdateDescriptorSets for the listed bindings.
    // descriptorType is looked up from the layout per binding.
    void update(const std::vector<DescriptorWrite>& writes);

    VkDescriptorSet get() const { return _ds; }
    const DescriptorSetLayout& layout() const { return *_layout; }

private:
    std::shared_ptr<VulkanContext> _ctx;
    std::shared_ptr<DescriptorSetLayout> _layout;
    VkDescriptorSet _ds = VK_NULL_HANDLE;
};
