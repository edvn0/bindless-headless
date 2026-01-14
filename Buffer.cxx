#include "Buffer.hxx"

#include "BindlessHeadless.hxx"

auto Buffer::set_name(VmaAllocator &allocator, std::string_view name) -> void {
    vmaSetAllocationName(allocator, allocation, name.data());
    set_debug_name(allocator, VK_OBJECT_TYPE_BUFFER, buffer, name);
}
