// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#endif

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_events.h"

namespace VideoCommon::Nx2Events {
namespace {

constexpr u32 MaxFrames = 12;
constexpr size_t TopStacks = 6;
constexpr size_t MaxNameLength = 100;
constexpr auto Interval = std::chrono::seconds{10};

struct Stack {
    std::array<void*, MaxFrames> frames{};
    u32 depth{};
    u64 count{};
};

struct Event {
    const char* name{};
    u64 count{};
    std::unordered_map<u64, Stack> stacks;
};

struct State {
    std::mutex mutex;
    std::vector<Event> events;
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};

/// Never destroyed: a report thread may still use it while the process exits.
State& GetState() {
    static State* const state = new State;
    return *state;
}

#ifdef _WIN32
/// Function names from DbgHelp, in a symbol session of its own (DbgHelp keeps one per process
/// handle value; the sampler uses GetCurrentProcess()'s).
class Symbols {
public:
    std::string Name(void* frame) {
        std::scoped_lock lock{mutex};
        const u64 address = reinterpret_cast<u64>(frame);
        const auto [it, is_new] = names.try_emplace(address);
        if (is_new) {
            it->second = Lookup(address);
        }
        return it->second;
    }

private:
    using SymInitializeFn = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
    using SymSetOptionsFn = DWORD(WINAPI*)(DWORD);
    using SymFromAddrFn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);

    std::string Lookup(u64 address) {
        Initialize();
        if (from_address != nullptr) {
            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512]{};
            auto* const symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 512;
            DWORD64 displacement{};
            if (from_address(process, address, &displacement, symbol)) {
                return std::string{symbol->Name,
                                   (std::min<size_t>)(symbol->NameLen, MaxNameLength)};
            }
        }
        HMODULE module{};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(address), &module)) {
            char path[MAX_PATH] = "?";
            GetModuleFileNameA(module, path, sizeof(path));
            const char* const base = std::strrchr(path, '\\');
            return fmt::format("{}+{:#x}", base != nullptr ? base + 1 : path,
                               address - reinterpret_cast<u64>(module));
        }
        return fmt::format("{:#x}", address);
    }

    void Initialize() {
        if (initialized) {
            return;
        }
        initialized = true;
        const HMODULE dbghelp = LoadLibraryA("dbghelp.dll");
        if (dbghelp == nullptr) {
            return;
        }
        const auto set_options =
            reinterpret_cast<SymSetOptionsFn>(GetProcAddress(dbghelp, "SymSetOptions"));
        const auto initialize =
            reinterpret_cast<SymInitializeFn>(GetProcAddress(dbghelp, "SymInitialize"));
        from_address = reinterpret_cast<SymFromAddrFn>(GetProcAddress(dbghelp, "SymFromAddr"));
        if (set_options == nullptr || initialize == nullptr || from_address == nullptr ||
            !DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), GetCurrentProcess(),
                             &process, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            from_address = nullptr;
            return;
        }
        set_options(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        if (!initialize(process, nullptr, TRUE)) {
            from_address = nullptr;
        }
    }

    std::mutex mutex;
    bool initialized{};
    HANDLE process{};
    SymFromAddrFn from_address{};
    std::unordered_map<u64, std::string> names;
};

Symbols& GetSymbols() {
    static Symbols* const symbols = new Symbols;
    return *symbols;
}
#endif

std::string FrameName([[maybe_unused]] void* frame) {
#ifdef _WIN32
    return GetSymbols().Name(frame);
#else
    return fmt::format("{}", frame);
#endif
}

void Report(std::vector<Event> events, double seconds) {
    std::ranges::sort(events, [](const Event& a, const Event& b) {
        return std::string_view{a.name} < std::string_view{b.name};
    });
    for (const Event& event : events) {
        LOG_INFO(HW_GPU, "{} in the last {:.1f} s: {}", event.name, seconds, event.count);
        std::vector<const Stack*> top;
        top.reserve(event.stacks.size());
        for (const auto& entry : event.stacks) {
            top.push_back(&entry.second);
        }
        const size_t shown = (std::min)(top.size(), TopStacks);
        std::partial_sort(top.begin(), top.begin() + shown, top.end(),
                          [](const Stack* a, const Stack* b) { return a->count > b->count; });
        for (size_t i = 0; i < shown; ++i) {
            std::string line;
            for (u32 frame = 0; frame < top[i]->depth; ++frame) {
                if (frame != 0) {
                    line += " <- ";
                }
                line += FrameName(top[i]->frames[frame]);
            }
            LOG_INFO(HW_GPU, "  {:7} ({:5.1f}%) {}", top[i]->count,
                     100.0 * static_cast<double>(top[i]->count) /
                         static_cast<double>(event.count),
                     line);
        }
    }
}

#ifdef _MSC_VER
__declspec(noinline)
#endif
void Record(const char* name, bool with_stack) {
    Stack captured{};
    u64 hash = 0;
#ifdef _WIN32
    if (with_stack) {
        ULONG frames_hash{};
        // Skips Record itself: the first frame is CountStack or (inlined) its caller.
        captured.depth = RtlCaptureStackBackTrace(1, MaxFrames, captured.frames.data(),
                                                  &frames_hash);
        hash = (u64{captured.depth} << 32) | frames_hash;
    }
#endif
    auto& state = GetState();
    std::unique_lock lock{state.mutex};
    auto it = std::ranges::find_if(state.events, [name](const Event& event) {
        return event.name == name || std::strcmp(event.name, name) == 0;
    });
    if (it == state.events.end()) {
        it = state.events.insert(state.events.end(), Event{.name = name});
    }
    ++it->count;
    if (captured.depth != 0) {
        Stack& stack = it->stacks[hash];
        if (stack.count == 0) {
            stack.frames = captured.frames;
            stack.depth = captured.depth;
        }
        ++stack.count;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - state.last_report < Interval) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.last_report).count();
    state.last_report = now;
    std::vector<Event> events = std::exchange(state.events, {});
    lock.unlock();
    // Looking names up loads the symbols the first time (seconds): not on the caller's thread.
    std::thread{[events = std::move(events), seconds]() mutable {
        Report(std::move(events), seconds);
    }}.detach();
}

} // Anonymous namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_EVENTS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void Count(const char* name) {
    Record(name, false);
}

void CountStack(const char* name) {
    Record(name, true);
}

} // namespace VideoCommon::Nx2Events
