// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "common/logging.h"
#include "core/arm/arm_interface.h"
#include "core/arm/debug.h"
#include "core/core.h"
#include "core/hle/kernel/k_memory_block.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"

namespace Core {

namespace {

// EDEN_NX2_PANIC_DUMP=<dir>: with each of the first four backtraces logged (a guest panic, abort or
// crash), save what is needed to read the failing check offline: the registers, the code, read-only
// data and data of every module on the backtrace, the top 64 KiB of the stack and 1 KiB around each
// register that points at mapped memory. panic-<n>.txt lists what was saved and where it was mapped.
void DumpGuestState(Kernel::KProcess* process, const Kernel::Svc::ThreadContext& ctx,
                    const std::vector<BacktraceEntry>& backtrace) {
    static const char* const dir = std::getenv("EDEN_NX2_PANIC_DUMP");
    static std::atomic<int> dumps{0};
    if (dir == nullptr || *dir == '\0') {
        return;
    }
    const int index = dumps.fetch_add(1);
    if (index >= 4) {
        return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto& memory = process->GetMemory();
    auto& page_table = process->GetPageTable();
    std::ofstream info{fs::path{dir} / fmt::format("panic-{}.txt", index)};
    info << fmt::format("pc {:016x} sp {:016x} lr {:016x} fp {:016x} tpidr {:016x}\n", ctx.pc,
                        ctx.sp, ctx.lr, ctx.fp, ctx.tpidr);
    for (size_t i = 0; i < ctx.r.size(); ++i) {
        info << fmt::format("x{} {:016x}\n", i, ctx.r[i]);
    }
    for (const auto& entry : backtrace) {
        info << fmt::format("frame {:016x}\n", entry.original_address);
    }
    const auto query = [&](u64 address, Kernel::KMemoryInfo& block) {
        Kernel::Svc::PageInfo page_info{};
        return page_table.QueryInfo(std::addressof(block), std::addressof(page_info), address)
            .IsSuccess();
    };
    const auto save = [&](u64 address, u64 size, const std::string& name) {
        if (size == 0 || size > (64ULL << 20) ||
            !memory.IsValidVirtualAddressRange(address, size)) {
            return;
        }
        std::vector<u8> bytes(size);
        memory.ReadBlock(address, bytes.data(), size);
        std::ofstream out{fs::path{dir} / name, std::ios::binary};
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        info << fmt::format("saved {} at {:016x} size {:x}\n", name, address, size);
    };
    std::set<u64> saved;
    for (const auto& entry : backtrace) {
        Kernel::KMemoryInfo block{};
        if (!query(entry.original_address, block)) {
            continue;
        }
        // A module's code, read-only data and data follow one another.
        u64 address = block.GetAddress();
        for (int segment = 0; segment < 3; ++segment) {
            Kernel::KMemoryInfo part{};
            if (!query(address, part) || part.GetPermission() == Kernel::KMemoryPermission::None) {
                break;
            }
            if (saved.insert(part.GetAddress()).second) {
                save(part.GetAddress(), part.GetSize(),
                     fmt::format("panic-{}-{:x}.bin", index, part.GetAddress()));
            }
            address = part.GetEndAddress();
        }
    }
    Kernel::KMemoryInfo stack{};
    if (query(ctx.sp, stack)) {
        save(ctx.sp, std::min<u64>(stack.GetEndAddress() - ctx.sp, 0x10000),
             fmt::format("panic-{}-stack-{:x}.bin", index, ctx.sp));
    }
    for (size_t i = 0; i < 31; ++i) {
        const u64 value = i < ctx.r.size() ? ctx.r[i] : (i == 29 ? ctx.fp : ctx.lr);
        const u64 start = (value & ~u64{0xF}) - 0x200;
        if (value > 0x200 && memory.IsValidVirtualAddressRange(start, 0x400)) {
            save(start, 0x400, fmt::format("panic-{}-x{}-{:x}.bin", index, i, start));
        }
    }
}

} // namespace

void ArmInterface::LogBacktrace(Kernel::KProcess* process) const {
    Kernel::Svc::ThreadContext ctx;
    this->GetContext(ctx);

    std::array<u64, 32> xreg{
        ctx.r[0], ctx.r[1], ctx.r[2], ctx.r[3],
        ctx.r[4], ctx.r[5], ctx.r[6], ctx.r[7],
        ctx.r[8], ctx.r[9], ctx.r[10], ctx.r[11],
        ctx.r[12], ctx.r[13], ctx.r[14], ctx.r[15],
        ctx.r[16], ctx.r[17], ctx.r[18], ctx.r[19],
        ctx.r[20], ctx.r[21], ctx.r[22], ctx.r[23],
        ctx.r[24], ctx.r[25], ctx.r[26], ctx.r[27],
        ctx.r[28], ctx.fp, ctx.lr, ctx.sp,
    };

    std::string msg = fmt::format("Backtrace @ PC={:016X}\n", ctx.pc);
    for (size_t i = 0; i < 32; i += 4)
        msg += fmt::format("R{:02}={:016X} R{:02}={:016X} R{:02}={:016X} R{:02}={:016X}\n",
            i + 0, xreg[i + 0], i + 1, xreg[i + 1],
            i + 2, xreg[i + 2], i + 3, xreg[i + 3]);
    for (size_t i = 0; i < 32; i += 2)
        msg += fmt::format("V{:02}={:016X}_{:016X} V{:02}={:016X}_{:016X}\n",
            i + 0, ctx.v[i + 0][0], ctx.v[i + 0][1],
            i + 1, ctx.v[i + 1][0], ctx.v[i + 1][1]);
    msg += fmt::format("PSTATE={:08X} FPCR={:08X} FPSR={:08X} TPIDR={:016X}\n", ctx.pstate, ctx.fpcr, ctx.fpsr, ctx.tpidr);
    msg += fmt::format("{:20}{:20}{:20}{:20}{}\n", "Module", "Address", "Original Address", "Offset", "Symbol");
    auto const backtrace = GetBacktraceFromContext(process, ctx);
    for (auto const& entry : backtrace)
        msg += fmt::format("{:20}{:016X}    {:016X}    {:016X}    {}\n", entry.module, entry.address, entry.original_address, entry.offset, entry.name);
    LOG_ERROR(Core_ARM, "{}", msg);
    DumpGuestState(process, ctx, backtrace);
}

const Kernel::DebugWatchpoint* ArmInterface::MatchingWatchpoint(
    u64 addr, u64 size, Kernel::DebugWatchpointType access_type) const {
    if (!m_watchpoints) {
        return nullptr;
    }

    const u64 start_address{addr};
    const u64 end_address{addr + size};

    for (size_t i = 0; i < Core::Hardware::NUM_WATCHPOINTS; i++) {
        const auto& watch{(*m_watchpoints)[i]};

        if (end_address <= GetInteger(watch.start_address)) {
            continue;
        }
        if (start_address >= GetInteger(watch.end_address)) {
            continue;
        }
        if ((access_type & watch.type) == Kernel::DebugWatchpointType::None) {
            continue;
        }

        return &watch;
    }

    return nullptr;
}

} // namespace Core
