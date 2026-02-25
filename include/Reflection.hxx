#pragma once

#include "ReflectionData.hxx"

#include <slang-com-ptr.h>
#include <slang.h>

auto reflect_program(Slang::ComPtr<slang::IComponentType> const &program, int target_index = 0) -> ReflectionData;
