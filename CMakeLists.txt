cmake_minimum_required(VERSION 3.16)
project(RadixSort)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)
    set(VULKAN_SDK $ENV{VULKAN_SDK})
    if (NOT VULKAN_SDK)
        message(FATAL_ERROR "Vulkan SDK not found! Make sure it's installed.")
    endif()
    set(CMAKE_INCLUDE_PATH ${VULKAN_SDK}/Include)
    set(CMAKE_LIBRARY_PATH ${VULKAN_SDK}/Lib)
endif()

find_package(Vulkan REQUIRED)
find_package(OpenMP REQUIRED)

# set the output directory for built objects.
# This makes sure that the dynamic library goes into the build directory automatically.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

# Add dependencies
add_subdirectory(external/spdlog)

# Libraries
set(LIBRARIES
    Vulkan::Vulkan
    spdlog::spdlog
    OpenMP::OpenMP_CXX
)

# Handle .cpp files
file(GLOB SOURCES
    src/bench/*.cpp
    src/utils/*.cpp
    src/vulkan/*.cpp
    src/vulkan/resources/*.cpp
)
list(APPEND SOURCES src/RadixSort.cpp)
list(APPEND SOURCES ${IMGUI_SOURCES})

# Add resources.rc file to the sources if on Windows
if (WIN32)
    list(APPEND SOURCES resources.rc)
endif()

# Handle .h files
set(INCLUDES ${INCLUDES} src)

# Set asset directory definition
add_compile_definitions("ASSET_DIR=${CMAKE_SOURCE_DIR}/assets/")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDES})
target_precompile_headers(${PROJECT_NAME} PRIVATE src/pch.h)
message(STATUS "I hate myself so I use cmake!")