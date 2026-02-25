#include "Types.hxx"

#define VULKAN_HPP_NO_TO_STRING
#include <vulkan/vulkan_to_string.hpp>
#include "vulkan/vulkan.hpp"

auto result_to_string(VkResult r) -> std::string {
    using namespace vk;
    Result res{r};
    return to_string(res);
}
