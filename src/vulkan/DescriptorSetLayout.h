#pragma once
#include "pch.h"
#include "VulkanContext.h"


class DescriptorSetLayout
{
public:
    struct Binding {
        uint32_t binding;
        VkDescriptorType type;
        uint32_t count = 1;
        VkShaderStageFlags stages;
    };

    DescriptorSetLayout(std::shared_ptr<VulkanContext> ctx,
                        const std::vector<Binding>& bindings);
    ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

    VkDescriptorSetLayout get() const { return _layout; }
    const std::vector<Binding>& getBindings() const { return _bindings; }
    VkDescriptorType typeOf(uint32_t binding) const;

private:
    std::shared_ptr<VulkanContext> _ctx;
    VkDescriptorSetLayout _layout = VK_NULL_HANDLE;
    std::vector<Binding> _bindings;
};
