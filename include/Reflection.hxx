#pragma once

#include "ReflectionData.hxx"

#if !defined(ENGINE_OFFLINE_SHADERS)

#include <slang-com-ptr.h>
#include <slang.h>

auto reflect_program(Slang::ComPtr<slang::IComponentType> const &program, int target_index = 0) -> ReflectionData;

#endif
