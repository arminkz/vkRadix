#pragma once

// [Vulkan]
#include <vulkan/vulkan.h>

// [SPDLOG]
#include "spdlog/spdlog.h"

// [Standard libraries: basic]
#include <cassert>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <execution>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <thread>
#include <filesystem>
#include <streambuf>
#include <optional>

// [Standard libraries: data structures]
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// [Time]
using TimePoint = std::chrono::high_resolution_clock::time_point;