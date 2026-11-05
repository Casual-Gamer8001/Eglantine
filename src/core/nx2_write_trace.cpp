// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/nx2_write_trace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Core::Nx2WriteTrace {
namespace {

constexpr std::size_t Capacity = std::size_t{1} << 18;
constexpr std::size_t MaxFrames = 8;
constexpr u32 MaxReports = 8;
constexpr u64 MaxScannedSpan = u64{64} << 20;
constexpr u64 MaxScannedTotal = u64{4} << 30;
constexpr std::size_t MaxMatches = 48;

struct Entry {
    u64 sequence; ///< Record number + 1; 0 while unused or being written
    s64 time_ns;
    u64 address;
    const u8* host;
    u64 size;
    u32 thread;
    u8 kind;
    u8 depth;
    void* frames[MaxFrames];
};

const char* Directory() {
    static const char* const directory = [] {
        const char* value = std::getenv("EDEN_NX2_WRITE_TRACE");
        return value != nullptr && *value != '\0' ? value : nullptr;
    }();
    return directory;
}

Entry* Entries() {
    static Entry* const entries = new Entry[Capacity]{};
    return entries;
}

std::atomic<u64> g_next{};

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const char* KindName(u8 kind) {
    switch (static_cast<Kind>(kind)) {
    case Kind::CpuWriteBlock:
        return "cpu WriteBlock";
    case Kind::DeviceWriteBlock:
        return "device WriteBlock";
    case Kind::DeviceWriteBlockUnsafe:
        return "device WriteBlockUnsafe";
    case Kind::GpuWriteBlock:
        return "gpu WriteBlock";
    }
    return "?";
}

#if defined(_WIN32) && defined(_MSC_VER)
// The span may have been unmapped since it was written: no C++ objects here (SEH).
bool SafeCopy(void* dest, const void* src, std::size_t size) {
    __try {
        std::memcpy(dest, src, size);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

std::pair<std::uintptr_t, std::uintptr_t> ExeRange() {
    const auto base = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    return {base, base + nt->OptionalHeader.SizeOfImage};
}
#else
bool SafeCopy(void* dest, const void* src, std::size_t size) {
    std::memcpy(dest, src, size);
    return true;
}

std::pair<std::uintptr_t, std::uintptr_t> ExeRange() {
    return {0, 0};
}
#endif

std::string FrameName(void* frame) {
    static const auto exe = ExeRange();
    const auto address = reinterpret_cast<std::uintptr_t>(frame);
    if (address >= exe.first && address < exe.second) {
        return fmt::format("exe+{:#x}", address - exe.first);
    }
    return fmt::format("{:#x}", address);
}

std::string Callers(const Entry& entry, std::size_t count) {
    std::string text;
    for (std::size_t i = 0; i < std::min<std::size_t>(entry.depth, count); ++i) {
        text += i == 0 ? "" : " < ";
        text += FrameName(entry.frames[i]);
    }
    return text;
}

struct Match {
    const Entry* entry;
    u64 offset; ///< Of the matching aligned word within the span
    u64 word;
};

} // Anonymous namespace

bool Enabled() {
    return Directory() != nullptr;
}

void Record(Kind kind, u64 address, const void* host, std::size_t size) {
    if (size == 0 || host == nullptr) {
        return;
    }
    Entry* const entries = Entries();
    const u64 number = g_next.fetch_add(1, std::memory_order_relaxed);
    Entry& entry = entries[number & (Capacity - 1)];
    entry.sequence = 0;
    entry.time_ns = NowNs();
    entry.address = address;
    entry.host = static_cast<const u8*>(host);
    entry.size = size;
    entry.kind = static_cast<u8>(kind);
#ifdef _WIN32
    entry.thread = GetCurrentThreadId();
    entry.depth = static_cast<u8>(
        RtlCaptureStackBackTrace(1, static_cast<DWORD>(MaxFrames), entry.frames, nullptr));
#else
    entry.thread = 0;
    entry.depth = 0;
#endif
    std::atomic_thread_fence(std::memory_order_release);
    entry.sequence = number + 1;
}

void OnGarbageAccess(u64 address, u64 value, std::size_t bytes, bool write) {
    static std::mutex mutex;
    static s64 last_report{};
    static u32 reports{};
    std::scoped_lock lock{mutex};
    const s64 now = NowNs();
    if (reports >= MaxReports || (last_report != 0 && now - last_report < 2'000'000'000)) {
        return;
    }
    last_report = now;
    const u32 report = ++reports;

    // Copy the ring, newest first.
    std::vector<Entry> ring(Entries(), Entries() + Capacity);
    std::vector<const Entry*> order;
    order.reserve(Capacity);
    for (const Entry& entry : ring) {
        if (entry.sequence != 0) {
            order.push_back(&entry);
        }
    }
    std::sort(order.begin(), order.end(),
              [](const Entry* a, const Entry* b) { return a->sequence > b->sequence; });

    // The pointer the guest followed is the garbage address minus a small field offset: words
    // whose upper 48 bits equal the address's are that pointer (or its source data).
    const u64 key = address >> 16;
    std::vector<Match> matches;
    std::vector<u8> chunk(std::size_t{1} << 20);
    u64 scanned{};
    std::size_t unreadable{};
    for (const Entry* entry : order) {
        if (matches.size() >= MaxMatches || scanned >= MaxScannedTotal) {
            break;
        }
        if (entry->size > MaxScannedSpan) {
            continue;
        }
        const auto host = reinterpret_cast<std::uintptr_t>(entry->host);
        const u64 first = (8 - (host & 7)) & 7;
        std::size_t found_here{};
        for (u64 done = 0; done < entry->size && found_here < 4;) {
            const u64 length = std::min<u64>(chunk.size(), entry->size - done);
            if (!SafeCopy(chunk.data(), entry->host + done, length)) {
                ++unreadable;
                break;
            }
            scanned += length;
            // Aligned words of the span (the chunk size keeps the alignment).
            for (u64 i = first; i + 8 <= length; i += 8) {
                u64 word;
                std::memcpy(&word, chunk.data() + i, 8);
                if ((word >> 16) == key) {
                    matches.push_back({entry, done + i, word});
                    if (++found_here >= 4) {
                        break;
                    }
                }
            }
            done += length;
        }
    }

    const std::string path = fmt::format("{}/nx2_write_trace-{}.txt", Directory(), report);
    std::FILE* const file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        LOG_ERROR(Core, "Write trace: cannot write {}", path);
        return;
    }
    const auto exe = ExeRange();
    const u64 newest = order.empty() ? 0 : order.front()->sequence;
    const s64 oldest_ns = order.empty() ? now : order.back()->time_ns;
    std::fprintf(file,
                 "Guest %s of %zu bytes at unmapped %#llx (value %#llx).\n"
                 "exe base %#llx size %#llx. Ring: %zu writes over the last %.2f s (%llu so far).\n"
                 "Scanned %.1f MiB of their spans for words %#llx????; %zu spans unreadable.\n\n",
                 write ? "write" : "read", bytes, static_cast<unsigned long long>(address),
                 static_cast<unsigned long long>(value), static_cast<unsigned long long>(exe.first),
                 static_cast<unsigned long long>(exe.second - exe.first), order.size(),
                 static_cast<double>(now - oldest_ns) / 1e9,
                 static_cast<unsigned long long>(newest), static_cast<double>(scanned) / 1048576.0,
                 static_cast<unsigned long long>(key), unreadable);

    std::fprintf(file, "== Spans that now hold the followed pointer (newest write first)\n");
    for (const Match& match : matches) {
        const Entry& entry = *match.entry;
        std::fprintf(file,
                     "%llu writes ago, %.3f ms ago: %s at %#llx (+%#llx of %#llx bytes) word "
                     "%#llx, host %p, thread %u\n    callers: %s\n",
                     static_cast<unsigned long long>(newest - entry.sequence),
                     static_cast<double>(now - entry.time_ns) / 1e6, KindName(entry.kind),
                     static_cast<unsigned long long>(entry.address),
                     static_cast<unsigned long long>(match.offset),
                     static_cast<unsigned long long>(entry.size),
                     static_cast<unsigned long long>(match.word),
                     static_cast<const void*>(entry.host + match.offset), entry.thread,
                     Callers(entry, MaxFrames).c_str());
        std::array<u8, 64> around{};
        const u64 start = match.offset >= 24 ? match.offset - 24 : 0;
        const u64 length = std::min<u64>(around.size(), entry.size - start);
        if (SafeCopy(around.data(), entry.host + start, length)) {
            std::fprintf(file, "    bytes from +%#llx:", static_cast<unsigned long long>(start));
            for (u64 i = 0; i < length; ++i) {
                std::fprintf(file, "%s%02x", i % 8 == 0 ? " " : "", around[i]);
            }
            std::fprintf(file, "\n");
        }
    }
    if (matches.empty()) {
        std::fprintf(file, "(none: the emulator's recorded writes don't hold it)\n");
    }

    // Who writes the most, for context.
    struct Total {
        u64 count{};
        u64 bytes{};
        u8 kind{};
    };
    std::map<std::string, Total> totals;
    for (const Entry* entry : order) {
        Total& total = totals[Callers(*entry, 4)];
        ++total.count;
        total.bytes += entry->size;
        total.kind = entry->kind;
    }
    std::vector<std::pair<std::string, Total>> sorted(totals.begin(), totals.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
    std::fprintf(file, "\n== Writers in the ring by bytes\n");
    for (std::size_t i = 0; i < std::min<std::size_t>(sorted.size(), 40); ++i) {
        std::fprintf(file, "%10llu bytes %7llu writes  %s  %s\n",
                     static_cast<unsigned long long>(sorted[i].second.bytes),
                     static_cast<unsigned long long>(sorted[i].second.count),
                     KindName(sorted[i].second.kind), sorted[i].first.c_str());
    }

    std::fprintf(file, "\n== Last 150 writes (newest first)\n");
    for (std::size_t i = 0; i < std::min<std::size_t>(order.size(), 150); ++i) {
        const Entry& entry = *order[i];
        std::fprintf(file, "%.3f ms ago: %s %#llx size %#llx host %p thread %u  %s\n",
                     static_cast<double>(now - entry.time_ns) / 1e6, KindName(entry.kind),
                     static_cast<unsigned long long>(entry.address),
                     static_cast<unsigned long long>(entry.size),
                     static_cast<const void*>(entry.host), entry.thread,
                     Callers(entry, 4).c_str());
    }
    std::fclose(file);
    LOG_WARNING(Core, "Write trace: guest {} at unmapped {:#x}; {} matching spans, report {}",
                write ? "write" : "read", address, matches.size(), path);
}

} // namespace Core::Nx2WriteTrace
