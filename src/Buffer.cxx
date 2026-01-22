#include "Buffer.hxx"

#include "BindlessHeadless.hxx"

auto Buffer::set_name(VmaAllocator &allocator, const std::string_view name) const -> void {
    vmaSetAllocationName(allocator, vma_allocation, name.data());
    set_debug_name(allocator, VK_OBJECT_TYPE_BUFFER, vk_buffer, name);
}
