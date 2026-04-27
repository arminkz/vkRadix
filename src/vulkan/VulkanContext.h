#pragma once
#include "pch.h"
#include "VulkanHelper.h"


struct VulkanContextConfig {
    std::string appName = "VulkanApp";
    uint32_t apiVersion = VK_API_VERSION_1_3;

    // Caller-supplied extensions.
    std::vector<const char*> instanceExtensions;
    std::vector<const char*> deviceExtensions;

    // Queue requirements. A presentation queue is implicitly required iff createSurface != nullptr.
    bool requireGraphicsQueue = false;
    bool requireComputeQueue  = false;

    // Headless / Compute Only: pass nullptr
    // Graphics: SDL_Vulkan_CreateSurface (or equivalent) in this lambda.
    std::function<VkSurfaceKHR(VkInstance)> createSurface = nullptr;

    // Device features.
    VkPhysicalDeviceFeatures features{};
    void* deviceCreateInfoPNext = nullptr;

    // Where pipeline_cache.bin is read/written.
    std::string basePath = "";

    bool preferDiscreteGPU = true;
    bool enableValidation = true;
};


class VulkanContext {
public:
    explicit VulkanContext(const VulkanContextConfig& config);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkInstance instance;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue computeQueue  = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue  = VK_NULL_HANDLE;

    std::optional<uint32_t> computeFamilyIndex;
    std::optional<uint32_t> graphicsFamilyIndex;
    std::optional<uint32_t> presentFamilyIndex;

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool;
    VkCommandPool commandPool;

    void printVulkanInfo();

private:
    VulkanContextConfig _config;
    bool _validationLayersAvailable = true;

    void createVulkanInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createDescriptorPool();
    void createCommandPool();

    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

    bool isDeviceSuitable(VkPhysicalDevice device, bool fallback = false);

    bool isInstanceLayerAvailable(const char* layerName);
    bool isInstanceExtensionAvailable(const char* extensionName);

    void loadPipelineCache(const std::string& filename);
    void savePipelineCache(const std::string& filename);
};
