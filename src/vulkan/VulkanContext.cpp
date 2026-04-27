#include "VulkanContext.h"

VulkanContext::VulkanContext(const VulkanContextConfig& config)
    : _config(config)
{
    createVulkanInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createDescriptorPool();
    createCommandPool();
    loadPipelineCache("pipeline_cache.bin");
}

VulkanContext::~VulkanContext() {
    savePipelineCache("pipeline_cache.bin");
    spdlog::info("Destroying Vulkan context...");
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipelineCache(device, pipelineCache, nullptr);
    vkDestroyDevice(device, nullptr);

    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    if (debugMessenger) {
        auto destroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroyDebugUtilsMessengerEXT != nullptr) {
            destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
    }

    vkDestroyInstance(instance, nullptr);
}

void VulkanContext::createVulkanInstance() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = _config.appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = _config.apiVersion;

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    bool enableVL = _config.enableValidation;

#if __APPLE__
    enableVL = false; // Disable validation layers on macOS for now (MoltenVK issues)
#endif

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    _validationLayersAvailable = isInstanceLayerAvailable("VK_LAYER_KHRONOS_validation");
    if (!_validationLayersAvailable || !enableVL) {
        spdlog::warn("Validation layers not available or disabled.");
        instanceCreateInfo.ppEnabledLayerNames = nullptr;
        instanceCreateInfo.enabledLayerCount = 0;
    } else {
        spdlog::info("Validation layers are available!");
        instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
        instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    }

    // Caller-supplied instance extensions (e.g. SDL surface extensions for graphics apps).
    std::vector<const char*> requiredExtensions(_config.instanceExtensions.begin(), _config.instanceExtensions.end());

    if (!isInstanceExtensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        spdlog::warn("Debug Utils extension not available!");
    } else {
        requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    if (!isInstanceExtensionAvailable(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        spdlog::warn("VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME not available!");
    } else {
        requiredExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = requiredExtensions.data();

#ifdef __APPLE__
    instanceCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    spdlog::debug("Enabled Vulkan Instance Extensions:");
    for (const char* extension : requiredExtensions) {
        spdlog::debug("  {}", extension);
    }

    VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan instance: {}", static_cast<int>(result));
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    spdlog::info("Vulkan instance created successfully");
}

void VulkanContext::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;

    auto createDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if (createDebugUtilsMessengerEXT != nullptr) {
        if (createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            spdlog::error("Failed to set up debug messenger!");
        } else {
            spdlog::info("Debug messenger set up successfully");
        }
    } else {
        spdlog::error("vkGetInstanceProcAddr failed to find vkCreateDebugUtilsMessengerEXT function!");
    }
}

void VulkanContext::createSurface() {
    if (!_config.createSurface) return; // headless / compute-only

    surface = _config.createSurface(instance);
    if (surface == VK_NULL_HANDLE) {
        throw std::runtime_error("createSurface callback returned VK_NULL_HANDLE");
    }
    spdlog::info("Vulkan surface created via caller-supplied callback");
}

void VulkanContext::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    if (_config.preferDiscreteGPU) {
        for (const auto& d : devices) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(d, &deviceProperties);
            if (isDeviceSuitable(d)) {
                physicalDevice = d;
                spdlog::info("Found Suitable dGPU: {}", deviceProperties.deviceName);
                break;
            }
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        if (_config.preferDiscreteGPU) {
            spdlog::warn("No suitable discrete GPU found, falling back to any suitable GPU.");
        }
        for (const auto& d : devices) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(d, &deviceProperties);
            if (isDeviceSuitable(d, true)) {
                physicalDevice = d;
                spdlog::info("Found Suitable iGPU: {}", deviceProperties.deviceName);
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }
}

void VulkanContext::createLogicalDevice() {
    QueueFamilyIndices indices = VulkanHelper::findQueueFamilies(physicalDevice, surface);
    computeFamilyIndex = indices.computeFamilyIndex;
    graphicsFamilyIndex = indices.graphicsFamilyIndex;
    presentFamilyIndex = indices.presentFamilyIndex;

    std::set<uint32_t> uniqueQueueFamilies;
    if (_config.requireComputeQueue) {
        if (!computeFamilyIndex.has_value()) throw std::runtime_error("No compute queue family found!");
        uniqueQueueFamilies.insert(computeFamilyIndex.value());
    }
    if (_config.requireGraphicsQueue) {
        if (!graphicsFamilyIndex.has_value()) throw std::runtime_error("No graphics queue family found!");
        uniqueQueueFamilies.insert(graphicsFamilyIndex.value());
    }
    if (surface != VK_NULL_HANDLE) {
        if (!presentFamilyIndex.has_value()) throw std::runtime_error("No present queue family found!");
        uniqueQueueFamilies.insert(presentFamilyIndex.value());
    }

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();

    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(_config.deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = _config.deviceExtensions.empty() ? nullptr : _config.deviceExtensions.data();

    deviceCreateInfo.pEnabledFeatures = &_config.features;
    deviceCreateInfo.pNext = _config.deviceCreateInfoPNext;

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    spdlog::info("Logical device created successfully");

    if (_config.requireComputeQueue) {
        vkGetDeviceQueue(device, computeFamilyIndex.value(), 0, &computeQueue);
    }
    if (_config.requireGraphicsQueue) {
        vkGetDeviceQueue(device, graphicsFamilyIndex.value(), 0, &graphicsQueue);
    }
    if (surface != VK_NULL_HANDLE) {
        vkGetDeviceQueue(device, presentFamilyIndex.value(), 0, &presentQueue);
    }
}

void VulkanContext::createDescriptorPool() {
    uint32_t totalUBOs = 100;
    uint32_t totalSamplers = 70;
    uint32_t totalSSBOs = 50;
    uint32_t maxSets = 100;

    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, totalUBOs },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, totalSamplers },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalSSBOs}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }

    spdlog::info("Descriptor pool created successfully");
}

void VulkanContext::createCommandPool() {
    // Pick a family that was actually requested at device creation time.
    // Prefer graphics if requested, otherwise compute.
    uint32_t familyIndex;
    if (_config.requireGraphicsQueue && graphicsFamilyIndex.has_value()) {
        familyIndex = graphicsFamilyIndex.value();
    } else if (_config.requireComputeQueue && computeFamilyIndex.has_value()) {
        familyIndex = computeFamilyIndex.value();
    } else {
        throw std::runtime_error("No requested queue family available for command pool!");
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = familyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }

    spdlog::info("Command pool created successfully");
}


bool VulkanContext::isDeviceSuitable(VkPhysicalDevice device, bool fallback) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    bool isDiscreteGPU = (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

    // Required device extensions come from the caller (e.g. swapchain for graphics apps).
    std::set<std::string> requiredExtensions(_config.deviceExtensions.begin(), _config.deviceExtensions.end());
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    bool hasRequiredExtensions = requiredExtensions.empty();

    // Swapchain adequacy only matters when a surface was provided.
    bool swapChainAdequate = true;
    if (surface != VK_NULL_HANDLE) {
        SwapChainSupportDetails swapChainSupport = VulkanHelper::querySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    // Queue family requirements per config.
    QueueFamilyIndices indices = VulkanHelper::findQueueFamilies(device, surface);
    bool queuesOk = true;
    if (_config.requireComputeQueue  && !indices.computeFamilyIndex.has_value())  queuesOk = false;
    if (_config.requireGraphicsQueue && !indices.graphicsFamilyIndex.has_value()) queuesOk = false;
    if (surface != VK_NULL_HANDLE    && !indices.presentFamilyIndex.has_value())  queuesOk = false;

    spdlog::debug("Evaluating GPU: {}", deviceProperties.deviceName);
    spdlog::debug(" - Discrete GPU: {}", isDiscreteGPU ? "Yes" : "No");
    spdlog::debug(" - Required Extensions: {}", hasRequiredExtensions ? "Yes" : "No");
    spdlog::debug(" - Swapchain Adequate: {}", swapChainAdequate ? "Yes" : "No");
    spdlog::debug(" - Queues Ok: {}", queuesOk ? "Yes" : "No");

    if (!fallback && _config.preferDiscreteGPU) {
        return isDiscreteGPU && hasRequiredExtensions && swapChainAdequate && queuesOk;
    }
    return hasRequiredExtensions && swapChainAdequate && queuesOk;
}

bool VulkanContext::isInstanceLayerAvailable(const char* layerName) {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const auto& layer : availableLayers) {
        if (strcmp(layerName, layer.layerName) == 0) {
            return true;
        }
    }
    return false;
}

bool VulkanContext::isInstanceExtensionAvailable(const char* extensionName) {
    uint32_t extensionCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

    for (const auto& extension : availableExtensions) {
        if (strcmp(extensionName, extension.extensionName) == 0) {
            return true;
        }
    }
    return false;
}


VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    spdlog::error("{}", pCallbackData->pMessage);

    return VK_FALSE;
}

void VulkanContext::loadPipelineCache(const std::string& filename) {
    std::string path = _config.basePath + filename;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::vector<char> cacheData;
    if (file.is_open()) {
        size_t size = static_cast<size_t>(file.tellg());
        cacheData.resize(size);
        file.seekg(0);
        file.read(cacheData.data(), size);
        file.close();

        VkPipelineCacheCreateInfo cacheCreateInfo{};
        cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        cacheCreateInfo.initialDataSize = size;
        cacheCreateInfo.pInitialData = cacheData.data();

        if (vkCreatePipelineCache(device, &cacheCreateInfo, nullptr, &pipelineCache) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline cache from file.");
        }
        spdlog::info("Pipeline cache loaded from file: {}", path);
    } else {
        spdlog::warn("No pipeline cache file found at {}, creating a new one.", path);
        VkPipelineCacheCreateInfo cacheCreateInfo{};
        cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        vkCreatePipelineCache(device, &cacheCreateInfo, nullptr, &pipelineCache);
    }
}

void VulkanContext::savePipelineCache(const std::string& filename) {
    size_t dataSize = 0;
    vkGetPipelineCacheData(device, pipelineCache, &dataSize, nullptr);
    std::vector<char> cacheData(dataSize);
    vkGetPipelineCacheData(device, pipelineCache, &dataSize, cacheData.data());

    std::string path = _config.basePath + filename;
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(cacheData.data(), dataSize);
        file.close();
        spdlog::info("Pipeline cache saved to file: {}", path);
    } else {
        spdlog::error("Failed to save pipeline cache to file: {}", path);
    }
}

void VulkanContext::printVulkanInfo() {
    spdlog::info("--------------------------------");
    spdlog::info("Vulkan API version: {}.{}",  VK_API_VERSION_MAJOR(_config.apiVersion), VK_API_VERSION_MINOR(_config.apiVersion));
    spdlog::info("--------------------------------");

    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    spdlog::info("Available layers:");
    for (const auto& layer : availableLayers) {
        spdlog::info("  {}", layer.layerName);
    }

    spdlog::info("------------------------------------");

    uint32_t extensionCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());
    spdlog::info("Available extensions:");
    for (const auto& extension : availableExtensions) {
        spdlog::info("  {}", extension.extensionName);
    }

    spdlog::info("------------------------------------");
    spdlog::info("Validation layers available: {}", _validationLayersAvailable ? "true" : "false");
}
