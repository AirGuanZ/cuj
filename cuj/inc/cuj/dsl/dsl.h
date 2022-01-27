#pragma once

#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/arithmetic_reference.h>
#include <cuj/dsl/array.h>
#include <cuj/dsl/array_reference.h>
#include <cuj/dsl/bitcast.h>
#include <cuj/dsl/class.h>
#include <cuj/dsl/const_data.h>
#include <cuj/dsl/exit_scope.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/global_var.h>
#include <cuj/dsl/if.h>
#include <cuj/dsl/inline_asm.h>
#include <cuj/dsl/loop.h>
#include <cuj/dsl/module.h>
#include <cuj/dsl/pointer.h>
#include <cuj/dsl/pointer_reference.h>
#include <cuj/dsl/reference.h>
#include <cuj/dsl/return.h>
#include <cuj/dsl/switch.h>
#include <cuj/dsl/type_context.h>
#include <cuj/dsl/variable.h>
#include <cuj/gen/gen.h>

#include <cuj/dsl/impl/arithmetic.inl>
#include <cuj/dsl/impl/arithmetic_reference.inl>
#include <cuj/dsl/impl/array.inl>
#include <cuj/dsl/impl/array_reference.inl>
#include <cuj/dsl/impl/bitcast.inl>
#include <cuj/dsl/impl/class.inl>
#include <cuj/dsl/impl/const_data.inl>
#include <cuj/dsl/impl/exit_scope.inl>
#include <cuj/dsl/impl/function.inl>
#include <cuj/dsl/impl/global_var.inl>
#include <cuj/dsl/impl/if.inl>
#include <cuj/dsl/impl/inline_asm.inl>
#include <cuj/dsl/impl/loop.inl>
#include <cuj/dsl/impl/pointer.inl>
#include <cuj/dsl/impl/pointer_reference.inl>
#include <cuj/dsl/impl/return.inl>
#include <cuj/dsl/impl/switch.inl>
#include <cuj/dsl/impl/type_context.inl>

CUJ_NAMESPACE_BEGIN(cuj)

using i8  = dsl::num<int8_t>;
using i16 = dsl::num<int16_t>;
using i32 = dsl::num<int32_t>;
using i64 = dsl::num<int64_t>;

using u8  = dsl::num<uint8_t>;
using u16 = dsl::num<uint16_t>;
using u32 = dsl::num<uint32_t>;
using u64 = dsl::num<uint64_t>;

using f32 = dsl::num<float>;
using f64 = dsl::num<double>;

using boolean = dsl::num<bool>;
using char_t = dsl::num<char>;

using dsl::Function;
using dsl::Module;
using dsl::ScopedModule;

using dsl::var;
using dsl::ref;

using dsl::ptr;
using dsl::num;
using dsl::arr;
using dsl::cxx;

using dsl::bitcast;
using dsl::const_data;
using dsl::string_literial;
using dsl::import_pointer;

using dsl::inline_asm;
using dsl::inline_asm_volatile;

using dsl::function;
using dsl::kernel;
using dsl::declare;

using dsl::allocate_global_memory;
using dsl::allocate_constant_memory;

CUJ_NAMESPACE_END(cuj)
