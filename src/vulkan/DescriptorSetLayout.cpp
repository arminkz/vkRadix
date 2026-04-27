#include "DescriptorSetLayout.h"


DescriptorSetLayout::DescriptorSetLayout(std::shared_ptr<VulkanContext> ctx,
                                         const std::vector<Binding>& bindings)
    : _ctx(std::move(ctx)), _bindings(bindings)
{
    std::vector<VkDescriptorSetLayoutBinding> vkBindings(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        vkBindings[i].binding            = bindings[i].binding;
        vkBindings[i].descriptorType     = bindings[i].type;
        vkBindings[i].descriptorCount    = bindings[i].count;
        vkBindings[i].stageFlags         = bindings[i].stages;
        vkBindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(vkBindings.size());
    ci.pBindings    = vkBindings.data();

    if (vkCreateDescriptorSetLayout(_ctx->device, &ci, nullptr, &_layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}


DescriptorSetLayout::~DescriptorSetLayout()
{
    if (_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(_ctx->device, _layout, nullptr);
    }
}


VkDescriptorType DescriptorSetLayout::typeOf(uint32_t binding) const
{
    for (const auto& b : _bindings) {
        if (b.binding == binding) return b.type;
    }
    throw std::runtime_error("DescriptorSetLayout::typeOf — binding not found");
}
