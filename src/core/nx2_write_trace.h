// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"

// EDEN_NX2_WRITE_TRACE=<directory>: finds which of the emulator's own writes into guest memory
// (HLE services, GPU downloads and copies) put garbage where the guest keeps its pointers. The
// recent writes are remembered (host span, guest or device address, size, caller); when the guest
// then follows a pointer into unmapped memory, the spans that still hold that pointer's bytes are
// reported to <directory>/nx2_write_trace-<n>.txt, with the callers that wrote them.
namespace Core::Nx2WriteTrace {

enum class Kind : u8 {
    CpuWriteBlock,          ///< Core::Memory::WriteBlock (guest virtual address)
    DeviceWriteBlock,       ///< DeviceMemoryManager::WriteBlock (device address)
    DeviceWriteBlockUnsafe, ///< DeviceMemoryManager::WriteBlockUnsafe (device address)
    GpuWriteBlock,          ///< Tegra::MemoryManager::WriteBlock* (GPU virtual address)
};

[[nodiscard]] bool Enabled();

/// A write of `size` bytes at `host` (guest or device `address`).
void Record(Kind kind, u64 address, const void* host, std::size_t size);

/// The guest accessed unmapped `address` (not the null page): reports the recent writes that hold
/// its pointer.
void OnGarbageAccess(u64 address, u64 value, std::size_t bytes, bool write);

} // namespace Core::Nx2WriteTrace
