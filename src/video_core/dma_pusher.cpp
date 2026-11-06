// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/nx2_profile.h"
#include "video_core/dma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 ComputeInline = 0x6D;
/// No engine class has methods past its macro calls (0xe00-0xfff).
constexpr u32 EngineMethodsEnd = 0x1000;
/// More registers than any state block one Increasing header writes.
constexpr u32 MaxIncreasingRun = 0x400;

/// EDEN_NX2_SPLIT_DUMP=<dir>: where splits across entries and misread submissions are written.
static const char* PushbufferDumpDirectory() {
    static const char* const directory = std::getenv("EDEN_NX2_SPLIT_DUMP");
    return directory;
}

DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

void DmaPusher::DispatchCalls() {
    // The sub-index is kept: a channel suspended on a semaphore resumes inside its command list.
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer([gpu = &system.GPU()] {
            gpu->FlushCommands();
            gpu->OnCommandListEnd();
        });
        split.Publish();
        return;
    }
    system.GPU().FlushCommands();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::Step() {
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }
    const u64 suspensions_before = suspensions.load(std::memory_order_relaxed);
    if (!puller.ResumeAcquire(*this)) {
        return false;
    }
    // Lists queued while the channel waited were read from memory, not as submitted: the game
    // writes the commands a list runs before it releases the acquire, so the words at submit time
    // could be older. Reading them as late as the parse is too late in Hyrule Warriors: Age of
    // Imprisonment fights: a command ring the game had already refilled for a later frame, read
    // as this frame's commands (methods on unbound subchannels, garbage QMDs, then a channel
    // waiting forever on a release it never got). The acquire is released now, so the commands
    // are written and the memory not yet reused: copy them here. EDEN_NX2_RESNAPSHOT=0: off.
    if (ResnapshotEnabled() && suspensions.load(std::memory_order_relaxed) != suspensions_before) {
        const u64 now = suspensions.load(std::memory_order_acquire);
        for (CommandList& list : dma_pushbuffer) {
            if (!list.submitted_offsets.empty()) {
                RefreshSnapshot(list, now, "after a semaphore wait");
            }
        }
    }

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset +=
            ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        dma_pushbuffer.pop_front();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (signal_sync && !synced) {
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced; });
        signal_sync = false;
        synced = false;
    }

    if (header.size > 0) {
        if (command_resume_offset == 0) {
            // NVN cuts a method's data across GP entries in two ways:
            // - the entry after the cut holds exactly the missing words (title 4: a macro's last
            //   parameters in a 5-word subroutine entry; Ocarina of Time: macro parameters the GPU
            //   writes) and the stream then goes on where the method's entry ended. The data
            //   continues into that entry, as on the hardware.
            // - the entry after the cut is another command buffer and a later entry starts where
            //   the method's entry ended, with the rest of the data (Kokiri Forest: an 11-word
            //   buffer runs between every 24-word piece of a list, cutting a viewport's depth range
            //   or a macro's parameters). Continuing there feeds the call's commands to the method
            //   and reads the rest of its data as commands, which ends in impossible headers and
            //   garbage engine binds. The method waits for the entry that continues it instead
            //   (ResolveSplitMethod) and the call starts clean.
            // Cuts whose next entry starts elsewhere are counted and, with EDEN_NX2_SPLIT_DUMP,
            // written out (DumpSplitAcrossEntries). EDEN_NX2_PARK_SPLIT_METHODS=0 always continues
            // and =1 always waits, for comparison.
            enum class SplitMode { Detect, Continue, Wait };
            static const SplitMode split_mode = [] {
                const char* value = std::getenv("EDEN_NX2_PARK_SPLIT_METHODS");
                if (value == nullptr || value[0] == '\0') {
                    return SplitMode::Detect;
                }
                return value[0] == '0' ? SplitMode::Continue : SplitMode::Wait;
            }();
            if (g_nx2_ampere) {
                bool park = false;
                if (dma_state.method_count != 0) {
                    if (u64{header.addr} == split_method_end) {
                        ++splits_continued;
                    } else {
                        ++splits_elsewhere;
                        DumpSplitAcrossEntries(command_list, dma_pushbuffer_subindex);
                        park = split_mode == SplitMode::Wait ||
                               (split_mode == SplitMode::Detect &&
                                IsCallInsideMethod(command_list, dma_pushbuffer_subindex));
                        if (park && split_mode == SplitMode::Detect) {
                            static std::atomic<u32> logged{};
                            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                                LOG_INFO(HW_GPU,
                                         "Method {:#x} (sub {}, {} words left) waits for the entry "
                                         "at {:#x}: entry {} at {:#x} ({} words) is a call",
                                         dma_state.method, dma_state.subchannel,
                                         dma_state.method_count, split_method_end,
                                         dma_pushbuffer_subindex, u64{header.addr},
                                         u64{header.size});
                            }
                        }
                    }
                }
                if (split_mode != SplitMode::Continue) {
                    ResolveSplitMethod(header.addr, park);
                }
            }
            if (dma_pushbuffer_subindex < command_list.submit_hashes.size()) {
                CheckSubmitHash(header, command_list.submit_hashes[dma_pushbuffer_subindex]);
            }
        }
        // After ResolveSplitMethod: a method set aside has no payload in this entry, and a resumed
        // one has (its macro parameters may be words the GPU wrote).
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            if (kepler_payload || macro_payload) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const engine_state = subchannels[dma_state.subchannel];
                if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
                    // The caches behind IsMemoryDirty belong to the split's back thread.
                    split.Defer([engine_state, memory = &memory_manager, address = GPUVAddr{dma_state.dma_get},
                                 bytes = words * sizeof(u32)] {
                        engine_state->current_dirty = memory->IsMemoryDirty(address, bytes);
                    });
                } else {
                    engine_state->current_dirty = memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                }
            }
        }
        // A resumed entry is read again: the words after the acquire may be new by now.
        // A safe read flushes the caches, which belong to the split's back thread while it runs
        // (the split only starts with unsafe reads; the settings can change mid-game).
        const bool use_safe = !VideoCommon::GpuSplit::Get().Active() &&
                              (Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe());
        // The split's back marks the pages GPU work writes as it runs that work: a page first
        // written by a dispatch or copy still queued for it reads as unwritten, and its words as
        // stale (Fast Fusion's culling fills in instanced draws' arguments; stale ones asked for
        // billions of instances and lost the device loading a race).
        bool gpu_written =
            IsGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
        if (!gpu_written && VideoCommon::GpuSplit::Get().Active() &&
            VideoCommon::GpuSplit::Get().WaitForWriters()) {
            gpu_written =
                IsGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
        }
        // The words as submitted, unless this is a resumed entry or the GPU has newer ones the
        // safe read below would flush (with the split, the front can't tell: it reads memory
        // unflushed anyway). Entries on pages the GPU writes (culling) always take the flushing
        // read: their words at submit time are an older frame's.
        const bool submitted = command_resume_offset == 0 &&
                               dma_pushbuffer_subindex < command_list.submitted_offsets.size() &&
                               (KeepSubmitCopies() ||
                                ((command_list.snapshot_suspensions & 1) == 0 &&
                                 command_list.snapshot_suspensions ==
                                     suspensions.load(std::memory_order_acquire))) &&
                               !gpu_written &&
                               !(use_safe && memory_manager.IsMemoryDirty(dma_state.dma_get,
                                                                          header.size * sizeof(u32)));
        if (submitted) {
            const std::span<const CommandHeader> commands{
                command_list.submitted_words.data() +
                    command_list.submitted_offsets[dma_pushbuffer_subindex],
                header.size};
            RecordParsedEntry(header, commands);
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (use_safe) {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else if (const auto read_start = GpuListStatsEnabled() && gpu_written
                                               ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
                   ReadGpuWrittenCommands(dma_state.dma_get, static_cast<u32>(header.size))) {
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            if (read_start != std::chrono::steady_clock::time_point{}) {
                NoteGpuWrittenList(dma_state.dma_get, commands, false,
                                   std::chrono::steady_clock::now() - read_start);
            }
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else {
            FlushGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            if (read_start != std::chrono::steady_clock::time_point{}) {
                NoteGpuWrittenList(dma_state.dma_get, commands, true,
                                   std::chrono::steady_clock::now() - read_start);
            }
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        }
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        if (dma_state.method_count != 0) {
            split_method_end = header.addr + header.size * sizeof(u32);
        }
    }

    if (++dma_pushbuffer_subindex >= command_list_size) {
        // A split method resumes within the submission that split it; ring addresses repeat
        // every frame, so a stale one could otherwise swallow the headers of a later entry.
        for (const auto& parked : parked_methods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 16) {
                LOG_WARNING(HW_GPU,
                            "Submission ended with parked method {:#x} (sub {}, {} words left, "
                            "resume {:#x})",
                            parked.method, parked.subchannel, parked.method_count,
                            parked.resume_address);
            }
        }
        parked_methods.clear();
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        // SignalFence invalidates the GPU caches, which can record copies into the scheduler: the
        // back's state, which a drained back still submits from in its idle hook.
        VideoCommon::GpuSplit::Get().Run([this] {
            rasterizer->SignalFence([this]() {
                std::scoped_lock lk(sync_mutex);
                synced = true;
                sync_cv.notify_all();
            });
        });
    }

    return true;
}

// EDEN_NX2_SEM_WATCH=<gpu address>: log every method whose argument is the low word of an
// address in the 16 bytes there (finds the engine and method that release a semaphore).
static const u32 g_semaphore_watch = [] {
    const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
    return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
}();

// EDEN_NX2_PB_CHECK also keeps the last parsed entries, for DumpCurrentEntry.
static const bool g_record_parsed_entries = std::getenv("EDEN_NX2_PB_CHECK") != nullptr;

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset) {
    for (size_t index = 0; index < commands.size();) {
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (g_nx2_ampere && dma_state.method < non_puller_methods) {
                max_write = 1; // a semaphore acquire may suspend the channel after any word
            }
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            // Fast-path batching for consecutive non-execution methods in increasing mode
            if (!dma_state.non_incrementing && !dma_increment_once &&
                dma_state.method >= non_puller_methods &&
                !VideoCommon::GpuSplit::Get().Active() &&
                g_semaphore_watch == 0) {
                auto subchannel = subchannels[dma_state.subchannel];
                if (subchannel != nullptr) {
                    const size_t available = std::min<size_t>(dma_state.method_count, commands.size() - index);
                    u32 batch_count = 0;
                    while (batch_count < available &&
                           (dma_state.method + batch_count) < subchannel->execution_mask.size() &&
                           !subchannel->execution_mask[dma_state.method + batch_count]) {
                        ++batch_count;
                    }
                    if (batch_count > 1) {
                        subchannel->method_sink.reserve(subchannel->method_sink.size() + batch_count);
                        for (u32 i = 0; i < batch_count; ++i) {
                            subchannel->method_sink.emplace_back(dma_state.method + i, commands[index + i].argument);
                        }
                        dma_state.method += batch_count;
                        dma_state.method_count -= batch_count;
                        index += batch_count;
                        continue;
                    }
                }
            }

            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            CallMethod(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            if (PushbufferDumpDirectory() != nullptr) [[unlikely]] {
                recent_headers[recent_header_count++ % recent_headers.size()] = {
                    .address = dma_state.dma_get + (word_offset + index) * sizeof(u32),
                    .word = command_header.argument,
                    .entry = static_cast<u32>(dma_pushbuffer_subindex),
                };
            }
            // No command active - this is the first word of a new one
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                if (SkipImpossibleHeader(command_header, word_offset + index)) [[unlikely]] {
                    break;
                }
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                if (SkipImpossibleHeader(command_header, word_offset + index)) [[unlikely]] {
                    break;
                }
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                if (SkipImpossibleHeader(command_header, word_offset + index)) [[unlikely]] {
                    break;
                }
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                if (SkipImpossibleHeader(command_header, word_offset + index)) [[unlikely]] {
                    break;
                }
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingAcquire()) {
            return index;
        }
    }
    return commands.size();
}

// NVN on Switch 2 ends GPFIFO entries in the middle of a method's data and calls another command
// buffer before the entry that carries the rest, which starts right where the first one ended.
// With park_current the entry at entry_address is such a call: the unfinished method, with a
// macro's parameters so far, is set aside so the call runs with a clean method state, and resumes
// when the stream continues where the method's entry ended.
void DmaPusher::ResolveSplitMethod(GPUVAddr entry_address, bool park_current) {
    const auto continued = std::ranges::find(parked_methods, entry_address,
                                             &ParkedMethod::resume_address);
    std::optional<ParkedMethod> resume;
    if (continued != parked_methods.end()) {
        resume = *continued;
        parked_methods.erase(continued);
    }
    if (park_current) {
        if (parked_methods.size() == MaxParkedMethods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 8) {
                const auto& oldest = parked_methods.front();
                LOG_WARNING(HW_GPU,
                            "Dropping parked method {:#x} (sub {}, {} words left, resume {:#x})",
                            oldest.method, oldest.subchannel, oldest.method_count,
                            oldest.resume_address);
            }
            parked_methods.erase(parked_methods.begin());
        }
        ParkedMethod parked{
            .method = dma_state.method,
            .subchannel = dma_state.subchannel,
            .method_count = dma_state.method_count,
            .non_incrementing = dma_state.non_incrementing,
            .increment_once = dma_increment_once,
            .resume_address = split_method_end,
        };
        // A macro's parameters so far wait with it, so the call's macros start clean. Left in
        // the engine, the call's 6-parameter macro ran the cut macro with both calls' parameters
        // and the rest started another call (Kokiri Forest: parameter FIFO underflows, a crash).
        if (dma_state.method >= MacroRegistersStart) {
            parked.macro_call = std::make_shared<Engines::MacroCall>();
            SwapEngineMacroCall(dma_state.subchannel, parked.macro_call);
        }
        parked_methods.push_back(std::move(parked));
        dma_state.method_count = 0;
        dma_increment_once = false;
    }
    if (resume && dma_state.method_count == 0) {
        dma_state.method = resume->method;
        dma_state.subchannel = resume->subchannel;
        dma_state.method_count = resume->method_count;
        dma_state.non_incrementing = resume->non_incrementing;
        dma_increment_once = resume->increment_once;
        if (resume->macro_call) {
            SwapEngineMacroCall(resume->subchannel, resume->macro_call);
        }
    }
}

void DmaPusher::SwapEngineMacroCall(u32 subchannel,
                                    const std::shared_ptr<Engines::MacroCall>& call) {
    auto* const engine = subchannels[subchannel];
    const auto type = subchannel_type[subchannel];
    if (engine == nullptr || (type != Engines::EngineTypes::Maxwell3D &&
                              type != Engines::EngineTypes::KeplerCompute)) {
        return;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        engine->SwapMacroCall(*call);
        return;
    }
    // Where CallMethod sends the parameters: the front's macros, else the back thread.
    if (split.FrontMacros()) {
        if (type == Engines::EngineTypes::Maxwell3D) {
            auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(engine);
            if (maxwell3d->GetNativeMacroEngine().native_turing) {
                maxwell3d->FrontSwapMacroCall(system, split, *call);
                return;
            }
        } else {
            auto* const compute = static_cast<Engines::KeplerCompute*>(engine);
            if (compute->FrontRunsMacros()) {
                compute->FrontSwapMacroCall(split, *call);
                return;
            }
        }
    }
    split.DeferFunction([engine, call] { engine->SwapMacroCall(*call); });
}

// The entry at index starts away from where the unfinished method's entry ended. Unless it holds
// exactly the missing words, it is a call, whether or not a later entry of the submission starts
// where the method's entry ended. At the start of gameplay Ocarina of Time also cuts methods that
// no later entry continues, and the entry after one starts with commands of its own. Continued
// into it, a cut macro took the header and first parameter of the entry's first macro call as
// its last parameters, and that call was lost. Set aside, the cut method is dropped when the
// submission ends.
bool DmaPusher::IsCallInsideMethod(const CommandList& list, size_t index) const {
    return u64{list.command_lists[index].size} != dma_state.method_count;
}

// An entry starts away from where the unfinished method's entry ended: it holds the missing words
// or is a call before them (IsCallInsideMethod). The first ones are logged;
// EDEN_NX2_SPLIT_DUMP=<dir> also writes the submission around each of the first 16 as
// <dir>/nx2_split-NN.txt: every GP entry (raw, level, privilege and sync bits, whether the GPU
// writes its pages, whether its words changed since they were submitted), the words of the
// entries around the split as submitted and as in memory now, and for kicked-off submissions
// whether the GPFIFO entries in guest memory still match the ones submitted.
void DmaPusher::DumpSplitAcrossEntries(const CommandList& list, size_t index) {
    const auto& entries = list.command_lists;
    const CommandListHeader& current = entries[index];
    static std::atomic<u32> events{};
    const u32 event = events.fetch_add(1, std::memory_order_relaxed);
    if (event < 64) {
        LOG_INFO(HW_GPU,
                 "Method {:#x} (sub {}, {} words left) unfinished at entry {} of {} (submission "
                 "{}, ch {}), which starts at {:#x}, not where the method's entry ended ({:#x}); "
                 "{} such splits, {} continued in place",
                 dma_state.method, dma_state.subchannel, dma_state.method_count, index,
                 entries.size(), list.submit_sequence, channel_state.bind_id, u64{current.addr},
                 split_method_end, splits_elsewhere, splits_continued);
    }
    const char* const directory = PushbufferDumpDirectory();
    if (directory == nullptr || event >= 16) {
        return;
    }
    const std::string path = fmt::format("{}/nx2_split-{:02}.txt", directory, event);
    std::FILE* const file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        return;
    }
    const auto put = [file](const std::string& line) {
        std::fputs(line.c_str(), file);
        std::fputc('\n', file);
    };
    // The entry whose end left the method unfinished, and the first one after that continues there.
    size_t unfinished = index > 0 ? index - 1 : 0;
    for (size_t i = index; i-- > 0;) {
        if (u64{entries[i].addr} + u64{entries[i].size} * sizeof(u32) == split_method_end) {
            unfinished = i;
            break;
        }
    }
    std::optional<size_t> resumes;
    for (size_t i = index + 1; i < entries.size() && i <= index + 64; ++i) {
        if (u64{entries[i].addr} == split_method_end) {
            resumes = i;
            break;
        }
    }
    put(fmt::format("split #{}: {}", event, DescribeSubmission(list)));
    put(fmt::format("pending: method {:#x} sub {} count {} non_incrementing {} increment_once {}",
                    dma_state.method, dma_state.subchannel, dma_state.method_count,
                    dma_state.non_incrementing, static_cast<bool>(dma_increment_once)));
    put(fmt::format("the method's entry {} ended at {:#x}; entry {} starts at {:#x}; {}",
                    unfinished, split_method_end, index, u64{current.addr},
                    resumes ? fmt::format("entry {} continues at {:#x}", *resumes, split_method_end)
                            : std::string{"no later entry continues there"}));
    put(fmt::format("splits so far: {} elsewhere, {} continued in place", splits_elsewhere,
                    splits_continued));
    std::vector<std::pair<size_t, const char*>> marks{
        {unfinished, "ended inside the method"},
        {index, "starts elsewhere"},
    };
    if (resumes) {
        marks.emplace_back(*resumes, "continues where the method's entry ended");
    }
    WriteSubmissionEntries(file, list, marks);
    // The words of the entries from the one that left the method unfinished to the continuation.
    const size_t first = unfinished > 0 ? unfinished - 1 : 0;
    WriteEntryWords(file, list, first,
                    (std::min)({entries.size() - 1, (resumes ? *resumes : index + 2) + 1,
                                first + 23}));
    std::fclose(file);
    LOG_INFO(HW_GPU, "Wrote {}", path);
}

std::string DmaPusher::DescribeSubmission(const CommandList& list) const {
    static constexpr std::array submit_kinds{"emulator-built", "SubmitGpfifo (entries in the ioctl)",
                                             "KickoffPb (entries in guest memory)",
                                             "SubmitGpfifo (inline input)"};
    const u64 now = suspensions.load(std::memory_order_acquire);
    const bool snapshot_valid =
        (list.snapshot_suspensions & 1) == 0 && list.snapshot_suspensions == now;
    std::string copies;
    if (list.recopies != 0) {
        copies = fmt::format("; copied again {} times, last {} {:.1f} ms after submit ({} entries "
                             "changed)",
                             list.recopies, list.recopy_reason, list.recopy_age_ms,
                             list.recopy_changed_entries);
    } else if (SnapshotTraceEnabled()) {
        copies = "; never copied again after submit";
    }
    return fmt::format(
        "channel {}, submission {} via {}, {} entries; snapshot {} (taken at suspension count {}, "
        "now {}){}",
        channel_state.bind_id, list.submit_sequence,
        submit_kinds[(std::min<size_t>)(list.submit_kind, submit_kinds.size() - 1)],
        list.command_lists.size(),
        list.submitted_offsets.empty() ? "none"
        : snapshot_valid               ? "valid"
                                       : "stale",
        list.snapshot_suspensions, now, copies);
}

void DmaPusher::WriteSubmissionEntries(std::FILE* file, const CommandList& list,
                                       std::span<const std::pair<size_t, const char*>> marks) {
    const auto put = [file](const std::string& line) {
        std::fputs(line.c_str(), file);
        std::fputc('\n', file);
    };
    const auto& entries = list.command_lists;
    // GPFIFO entries kicked off from guest memory: are they still what was submitted?
    if (list.gpfifo_address != 0) {
        std::vector<u64> ring(entries.size());
        system.ApplicationMemory().ReadBlock(list.gpfifo_address, ring.data(),
                                             ring.size() * sizeof(u64));
        size_t changed = 0;
        std::string first_changes;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (ring[i] != entries[i].raw) {
                if (++changed <= 16) {
                    first_changes += fmt::format(" [{}] {:#018x}->{:#018x}", i, entries[i].raw, ring[i]);
                }
            }
        }
        bool ring_gpu_written = false;
        if (list.gpfifo_device_address != 0) {
            const u64 bytes = entries.size() * sizeof(u64);
            for (u64 page = list.gpfifo_device_address & ~u64{0xfff};
                 page < list.gpfifo_device_address + bytes; page += 0x1000) {
                ring_gpu_written |= Nx2IsGpuWritten(page);
            }
        }
        put(fmt::format("GPFIFO at {:#x} (device {:#x}, GPU writes its pages: {}): {} of {} entries "
                        "differ in guest memory now{}",
                        list.gpfifo_address, list.gpfifo_device_address, ring_gpu_written, changed,
                        entries.size(), first_changes));
    }
    // Every GP entry. "words" compares the submitted copy with guest memory now.
    put(fmt::format("GP entries: index raw address words level(1=subroutine) priv sync gpu-written "
                    "words{}",
                    list.submit_copy.empty() ? "" : " submit-copy"));
    std::vector<u32> memory_words;
    for (size_t i = 0; i < entries.size(); ++i) {
        const CommandListHeader& entry = entries[i];
        std::string words_state = "-";
        if (i < list.submitted_offsets.size() && entry.size <= 0x10000 &&
            list.submitted_offsets[i] + entry.size <= list.submitted_words.size()) {
            memory_words.resize(entry.size);
            memory_manager.ReadBlockUnsafe(entry.addr, memory_words.data(),
                                           memory_words.size() * sizeof(u32));
            const auto* submitted = list.submitted_words.data() + list.submitted_offsets[i];
            words_state = std::equal(memory_words.begin(), memory_words.end(), submitted,
                                     [](u32 word, const CommandHeader& header) {
                                         return word == header.argument;
                                     })
                              ? "same"
                              : "CHANGED";
        }
        // The executed copy against the words as submitted (EDEN_NX2_SNAPSHOT_TRACE).
        if (!list.submit_copy.empty() && i < list.submitted_offsets.size() &&
            list.submitted_offsets[i] + entry.size <= list.submit_copy.size() &&
            list.submitted_offsets[i] + entry.size <= list.submitted_words.size()) {
            const auto* executed = list.submitted_words.data() + list.submitted_offsets[i];
            const auto* at_submit = list.submit_copy.data() + list.submitted_offsets[i];
            words_state += std::equal(executed, executed + size_t{entry.size}, at_submit,
                                      [](const CommandHeader& a, const CommandHeader& b) {
                                          return a.argument == b.argument;
                                      })
                               ? " same"
                               : " RECOPIED";
        }
        const bool gpu_written = IsGpuWrittenCommands(entry.addr, size_t{entry.size} * sizeof(u32));
        std::string mark;
        for (const auto& [marked, text] : marks) {
            if (marked == i) {
                mark += fmt::format("  <- {}", text);
            }
        }
        put(fmt::format("  [{:4}] {:#018x} {:#012x} {:5} {} {} {} {:5} {}{}", i, entry.raw,
                        u64{entry.addr}, u64{entry.size}, u64{entry.is_push_buffer},
                        u64{entry.allow_flush}, u64{entry.sync}, gpu_written, words_state, mark));
    }
}

void DmaPusher::WriteEntryWords(std::FILE* file, const CommandList& list, size_t first,
                                size_t last) {
    const auto put = [file](const std::string& line) {
        std::fputs(line.c_str(), file);
        std::fputc('\n', file);
    };
    const auto& entries = list.command_lists;
    std::vector<u32> memory_words;
    for (size_t i = first; i <= last && i < entries.size(); ++i) {
        const CommandListHeader& entry = entries[i];
        const size_t count = (std::min<size_t>)(entry.size, 512);
        put(fmt::format("entry {} at {:#x}, {} words{}:", i, u64{entry.addr}, u64{entry.size},
                        count < entry.size ? " (first 512)" : ""));
        memory_words.resize(count);
        memory_manager.ReadBlockUnsafe(entry.addr, memory_words.data(), count * sizeof(u32));
        const bool has_submitted = i < list.submitted_offsets.size() &&
                                   list.submitted_offsets[i] + entry.size <=
                                       list.submitted_words.size();
        for (size_t word = 0; word < count; word += 8) {
            std::string submitted_line;
            std::string memory_line;
            for (size_t j = word; j < (std::min)(count, word + 8); ++j) {
                const u32 now = memory_words[j];
                if (has_submitted) {
                    const u32 then = list.submitted_words[list.submitted_offsets[i] + j].argument;
                    submitted_line += fmt::format(" {:08x}", then);
                    memory_line += then == now ? std::string(" ........") : fmt::format(" {:08x}", now);
                } else {
                    memory_line += fmt::format(" {:08x}", now);
                }
            }
            if (has_submitted) {
                put(fmt::format("  +{:03x} submitted:{}", word * 4, submitted_line));
                put(fmt::format("       memory   :{}", memory_line));
                // The words copied at submit, where a later copy replaced them.
                if (list.submitted_offsets[i] + entry.size <= list.submit_copy.size()) {
                    std::string submit_line;
                    bool differs = false;
                    for (size_t j = word; j < (std::min)(count, word + 8); ++j) {
                        const size_t at = list.submitted_offsets[i] + j;
                        const u32 then = list.submit_copy[at].argument;
                        const bool same = then == list.submitted_words[at].argument;
                        differs |= !same;
                        submit_line +=
                            same ? std::string(" ........") : fmt::format(" {:08x}", then);
                    }
                    if (differs) {
                        put(fmt::format("       at submit:{}", submit_line));
                    }
                }
            } else {
                put(fmt::format("  +{:03x} memory:{}", word * 4, memory_line));
            }
        }
    }
}

// EDEN_NX2_SPLIT_DUMP=<dir>: the first 8 impossible headers each write <dir>/nx2_misread-NN.txt:
// the header and where it was, the last 64 headers parsed before it (how the parse got there),
// every GP entry of the submission and the words of the 24 entries up to it and the 4 after.
void DmaPusher::DumpMisreadSubmission(const CommandList& list, const CommandHeader& header,
                                      size_t word_index, u32 event) {
    const std::string path =
        fmt::format("{}/nx2_misread-{:02}.txt", PushbufferDumpDirectory(), event);
    std::FILE* const file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        return;
    }
    const auto put = [file](const std::string& line) {
        std::fputs(line.c_str(), file);
        std::fputc('\n', file);
    };
    const size_t index = dma_pushbuffer_subindex;
    put(fmt::format("misread #{}: {}", event, DescribeSubmission(list)));
    put(fmt::format("header {:#010x} (mode {}, sub {}, method {:#x}, count {}) at word {} of "
                    "entry {} ({:#x}, {} words); resumed at word {}",
                    header.argument, static_cast<u32>(header.mode.Value()),
                    header.subchannel.Value(), header.method.Value(), header.method_count.Value(),
                    word_index, index, dma_state.dma_get,
                    index < list.command_lists.size() ? u64{list.command_lists[index].size} : 0,
                    command_resume_offset));
    put(fmt::format("splits so far: {} elsewhere, {} continued in place; last unfinished entry "
                    "ended at {:#x}",
                    splits_elsewhere, splits_continued, split_method_end));
    put("last headers parsed (oldest first): address word entry");
    const u64 recorded = (std::min<u64>)(recent_header_count, recent_headers.size());
    for (u64 i = recent_header_count - recorded; i < recent_header_count; ++i) {
        const ParsedHeader& parsed = recent_headers[i % recent_headers.size()];
        CommandHeader decoded{};
        decoded.argument = parsed.word;
        put(fmt::format("  {:#012x} {:08x} entry {:4}  mode {} sub {} method {:#x} count {}",
                        parsed.address, parsed.word, parsed.entry,
                        static_cast<u32>(decoded.mode.Value()), decoded.subchannel.Value(),
                        decoded.method.Value(), decoded.method_count.Value()));
    }
    WriteSubmissionEntries(file, list, std::array{std::pair<size_t, const char*>{index, "misread here"}});
    WriteEntryWords(file, list, index > 24 ? index - 24 : 0, index + 4);
    std::fclose(file);
    LOG_INFO(HW_GPU, "Wrote {}", path);
}
bool DmaPusher::GpuListStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_LIST_STATS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

namespace {
/// Which words of a command list are instructions, parsing it from a clean method state.
void MarkInstructions(std::span<const u32> words, std::vector<bool>& instructions) {
    instructions.assign(words.size(), false);
    u32 data_left = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        if (data_left != 0) {
            --data_left;
            continue;
        }
        instructions[i] = true;
        CommandHeader header{};
        header.argument = words[i];
        if (header.mode == SubmissionMode::Increasing ||
            header.mode == SubmissionMode::NonIncreasing ||
            header.mode == SubmissionMode::IncreaseOnce) {
            data_left = header.method_count;
        }
    }
}
} // Anonymous namespace

// EDEN_NX2_GPU_LIST_STATS=1: every 10 s, how the GPU thread read the command lists the GPU writes
// (from the copy the buffer cache made, after waiting for it, or through a cache flush), the time
// that took, and what changed since the same list was last read: nothing, only method data (the
// arguments the culling fills in) or instructions too. With EDEN_NX2_SPLIT_DUMP=<dir>, the first
// changes of each kind are written to <dir>/nx2_gpulist-{data,instructions}-N.txt.
void DmaPusher::NoteGpuWrittenList(GPUVAddr address, std::span<const CommandHeader> words,
                                   bool flushed, std::chrono::steady_clock::duration took) {
    auto& stats = gpu_list_stats;
    const u64 ns =
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(took).count());
    ++stats.reads;
    stats.flushed += flushed ? 1 : 0;
    stats.slow += ns >= 100'000 ? 1 : 0;
    stats.read_ns += ns;
    stats.max_ns = (std::max)(stats.max_ns, ns);
    stats.words += words.size();
    if (gpu_list_history.size() >= 8192) {
        gpu_list_history.clear();
    }
    std::vector<u32>& last = gpu_list_history[address];
    std::vector<u32> now(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        now[i] = words[i].argument;
    }
    if (last.empty()) {
        ++stats.first;
    } else if (last.size() != now.size()) {
        ++stats.resized;
    } else {
        std::vector<bool> old_instructions;
        std::vector<bool> new_instructions;
        MarkInstructions(last, old_instructions);
        MarkInstructions(now, new_instructions);
        bool data_changed = false;
        bool instructions_changed = false;
        for (size_t i = 0; i < now.size(); ++i) {
            if (now[i] == last[i] && old_instructions[i] == new_instructions[i]) {
                continue;
            }
            if (old_instructions[i] || new_instructions[i]) {
                instructions_changed = true;
            } else {
                data_changed = true;
            }
        }
        if (instructions_changed) {
            ++stats.headers_changed;
        } else if (data_changed) {
            ++stats.data_changed;
        } else {
            ++stats.unchanged;
        }
        static const char* const directory = std::getenv("EDEN_NX2_SPLIT_DUMP");
        static std::atomic<u32> data_dumps{};
        static std::atomic<u32> instruction_dumps{};
        if (directory != nullptr && (instructions_changed || data_changed)) {
            const u32 number = (instructions_changed ? instruction_dumps : data_dumps).fetch_add(1);
            if (number < 6) {
                const std::string path =
                    fmt::format("{}/nx2_gpulist-{}-{}.txt", directory,
                                instructions_changed ? "instructions" : "data", number);
                if (std::FILE* const file = std::fopen(path.c_str(), "w")) {
                    std::fputs(fmt::format("list at {:#x}, {} words, read {} in {} us\n", address,
                                           now.size(), flushed ? "through a flush" : "from a copy",
                                           ns / 1000)
                                   .c_str(),
                               file);
                    std::fputs("word: before -> now (I = instruction, . = data)\n", file);
                    for (size_t i = 0; i < (std::min<size_t>)(now.size(), 2048); ++i) {
                        std::fputs(fmt::format("  {:4}: {:08x} {} -> {:08x} {}{}\n", i, last[i],
                                               old_instructions[i] ? 'I' : '.', now[i],
                                               new_instructions[i] ? 'I' : '.',
                                               last[i] != now[i] ? "  *" : "")
                                       .c_str(),
                                   file);
                    }
                    std::fclose(file);
                }
            }
        }
    }
    last = std::move(now);
    const auto current = std::chrono::steady_clock::now();
    if (gpu_list_stats_since == std::chrono::steady_clock::time_point{}) {
        gpu_list_stats_since = current;
    }
    const double seconds = std::chrono::duration<double>(current - gpu_list_stats_since).count();
    if (seconds >= 10.0) {
        LOG_INFO(HW_GPU,
                 "GPU-written command lists in the last {:.1f} s: {} read ({} words), {} through a "
                 "cache flush, {} took over 0.1 ms; {:.1f} ms reading, longest {:.2f} ms. Since "
                 "their last read: {} new, {} resized, {} unchanged, {} only data changed, {} "
                 "instructions changed",
                 seconds, stats.reads, stats.words, stats.flushed, stats.slow,
                 static_cast<double>(stats.read_ns) / 1e6, static_cast<double>(stats.max_ns) / 1e6,
                 stats.first, stats.resized, stats.unchanged, stats.data_changed,
                 stats.headers_changed);
        stats = {};
        gpu_list_stats_since = current;
    }
}

bool DmaPusher::IsImpossibleHostSequence(u32 first, u32 last) {
    if (first >= non_puller_methods) {
        return false;
    }
    return (first <= 1 && last >= 1) || last >= non_puller_methods;
}

bool DmaPusher::IsImpossibleEngineSequence(u32 first, u32 last, u32 run) {
    if (last >= EngineMethodsEnd || run > MaxIncreasingRun) {
        return true;
    }
    return first < MacroRegistersStart && last >= MacroRegistersStart;
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}

// Only a misread stream holds such a header; taken alone as one word, the parse can fall back
// into step when the words after it are headers. The one in Kokiri Forest (Increasing, host
// method 0, 8064 words) would have written the commands after it over every register. Twice
// more there, a misread stream on the 3D engine reached method 0x1a52 (past its last method)
// and wrote the commands after it over render targets, transform feedback and constant buffers;
// building a pipeline for the next draw then read past the transform feedback layouts.
bool DmaPusher::SkipImpossibleHeader(const CommandHeader& command_header, size_t word_index) {
    const u32 method = command_header.method;
    const bool host = method < non_puller_methods;
    bool impossible;
    u32 count;
    if (command_header.mode == SubmissionMode::Inline) {
        // One write of the header's own value.
        count = 1;
        impossible = !host && method >= EngineMethodsEnd;
    } else {
        count = command_header.method_count;
        if (count == 0) {
            return false;
        }
        const bool increasing = command_header.mode == SubmissionMode::Increasing;
        const u32 last = increasing ? method + count - 1
                         : command_header.mode == SubmissionMode::IncreaseOnce
                             ? method + (count > 1 ? 1 : 0)
                             : method;
        impossible = host ? IsImpossibleHostSequence(method, last)
                          : IsImpossibleEngineSequence(method, last, increasing ? count : 1);
    }
    if (!impossible) {
        return false;
    }
    static std::atomic<u32> reported{};
    const u32 event = reported.fetch_add(1);
    if (event < 16) {
        LOG_WARNING(HW_GPU,
                    "Skipping impossible header {:#010x} ({} method {:#x}, {} words) on ch {} "
                    "in entry {} at {:#x}, word {} (submission {})",
                    command_header.argument, host ? "host" : "engine", method, count,
                    channel_state.bind_id, dma_pushbuffer_subindex, dma_state.dma_get, word_index,
                    dma_pushbuffer.empty() ? u64{0} : dma_pushbuffer.front().submit_sequence);
    }
    if (event < 8 && PushbufferDumpDirectory() != nullptr && !dma_pushbuffer.empty()) {
        DumpMisreadSubmission(dma_pushbuffer.front(), command_header, word_index, event);
    }
    return true;
}


void DmaPusher::WatchArgument(u32 method, u32 argument) const {
    const u32 watch = g_semaphore_watch;
    if ((argument & ~0xfU) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} subchannel {} method {:#x} argument {:#x}",
                 channel_state.bind_id, dma_state.subchannel, method, argument);
    }
}

bool DmaPusher::WantsSegment() const {
    // Only macro parameters and compute uploads read the engine's current_dma_segment.
    const auto engine = subchannel_type[dma_state.subchannel];
    return engine == Engines::EngineTypes::KeplerCompute ||
           (engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart);
}

void DmaPusher::CallMethod(u32 argument) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        WatchArgument(dma_state.method, argument);
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        // Compute dispatches and DMA copies write the command lists ReadGpuWrittenCommands reads.
        const auto engine_type = subchannel_type[dma_state.subchannel];
        const bool writer = engine_type == Engines::EngineTypes::KeplerCompute ||
                            engine_type == Engines::EngineTypes::MaxwellDMA;
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMethod(system, split, dma_state.method, argument,
                                                   dma_state.is_last_call);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMethod(system, split, dma_state.method, argument,
                                                 dma_state.is_last_call, &segment);
                        return;
                    }
                }
            }
            split.Method(subchannel, dma_state.method, argument, dma_state.is_last_call,
                         WantsSegment() ? &segment : nullptr);
            return;
        }
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        for (u32 i = 0; i < num_methods; ++i) {
            WatchArgument(dma_state.method + (dma_state.non_incrementing ? 0 : i), base_start[i]);
        }
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        // Compute dispatches and DMA copies write the command lists ReadGpuWrittenCommands reads.
        const auto engine_type = subchannel_type[dma_state.subchannel];
        const bool writer = engine_type == Engines::EngineTypes::KeplerCompute ||
                            engine_type == Engines::EngineTypes::MaxwellDMA;
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                        num_methods, dma_state.method_count);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                      num_methods, dma_state.method_count, &segment);
                        return;
                    }
                }
            }
            split.MultiMethod(subchannel, dma_state.method, base_start, num_methods,
                              dma_state.method_count, WantsSegment() ? &segment : nullptr);
            return;
        }
        VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

void DmaPusher::RefreshSnapshot(CommandList& list, u64 now, const char* reason) {
    const u64 old_count = list.snapshot_suspensions;
    list.snapshot_suspensions = now;
    const bool trace = SnapshotTraceEnabled();
    u32 changed_entries = 0;
    u64 changed_words = 0;
    size_t first_changed = list.command_lists.size();
    for (size_t i = 0; i < list.command_lists.size() && i < list.submitted_offsets.size(); ++i) {
        const CommandListHeader& entry = list.command_lists[i];
        const u32 offset = list.submitted_offsets[i];
        if (offset + entry.size > list.submitted_words.size()) {
            continue;
        }
        CommandHeader* const words = list.submitted_words.data() + offset;
        if (!trace) {
            memory_manager.ReadBlockUnsafe(entry.addr, words, entry.size * sizeof(u32));
            continue;
        }
        recopy_scratch.resize(entry.size);
        memory_manager.ReadBlockUnsafe(entry.addr, recopy_scratch.data(), entry.size * sizeof(u32));
        u32 differ = 0;
        for (u32 j = 0; j < entry.size; ++j) {
            differ += words[j].argument != recopy_scratch[j] ? 1 : 0;
        }
        if (differ != 0) {
            ++changed_entries;
            changed_words += differ;
            first_changed = (std::min)(first_changed, i);
            std::memcpy(words, recopy_scratch.data(), entry.size * sizeof(u32));
        }
    }
    if (trace) {
        NoteRecopy(list, reason, old_count, now, changed_entries, changed_words, first_changed);
    }
}

void DmaPusher::NoteRecopy(CommandList& list, const char* reason, u64 old_count, u64 now,
                           u32 changed_entries, u64 changed_words, size_t first_changed) {
    const auto time = std::chrono::steady_clock::now();
    const float age_ms =
        list.submitted_at == std::chrono::steady_clock::time_point{}
            ? -1.0f
            : std::chrono::duration<float, std::milli>(time - list.submitted_at).count();
    ++list.recopies;
    list.recopy_changed_entries = changed_entries;
    list.recopy_age_ms = age_ms;
    list.recopy_reason = reason;
    if (static std::atomic<u32> logged{};
        changed_entries != 0 && logged.fetch_add(1, std::memory_order_relaxed) < 400) {
        LOG_INFO(HW_GPU,
                 "Channel {} copied submission {} again {}, {:.1f} ms after it was submitted "
                 "(suspension count {} -> {}): {} of {} entries ({} words) changed, the first "
                 "entry {} at {:#x}",
                 channel_state.bind_id, list.submit_sequence, reason, age_ms, old_count, now,
                 changed_entries, list.command_lists.size(), changed_words, first_changed,
                 u64{list.command_lists[first_changed].addr});
    }
    // Every 10 s (GPU thread): how many copies were taken again, how late, and how many changed.
    struct Stats {
        u64 copies;
        u64 changed;
        u64 changed_entries;
        float max_age_ms;
        float max_changed_age_ms;
        std::chrono::steady_clock::time_point since;
    };
    thread_local Stats stats{};
    if (stats.since == std::chrono::steady_clock::time_point{}) {
        stats.since = time;
    }
    ++stats.copies;
    stats.max_age_ms = (std::max)(stats.max_age_ms, age_ms);
    if (changed_entries != 0) {
        ++stats.changed;
        stats.changed_entries += changed_entries;
        stats.max_changed_age_ms = (std::max)(stats.max_changed_age_ms, age_ms);
    }
    const float seconds = std::chrono::duration<float>(time - stats.since).count();
    if (seconds >= 10.0f) {
        LOG_INFO(HW_GPU,
                 "Command lists copied again in {:.0f} s: {} (latest {:.1f} ms after submit), {} "
                 "changed {} entries (latest {:.1f} ms after submit)",
                 seconds, stats.copies, stats.max_age_ms, stats.changed, stats.changed_entries,
                 stats.max_changed_age_ms);
        stats = Stats{};
        stats.since = time;
    }
}

bool DmaPusher::IsGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            return true;
        }
    }
    return false;
}

bool DmaPusher::ReadGpuWrittenCommands(GPUVAddr address, u32 count) {
    // The pages are marked as command list pages: from the next write on, the buffer cache copies
    // them to host memory right after the dispatch or copy that writes them and submits. The
    // front then waits only for the back to run that work and for the GPU to finish the copy, not
    // for everything queued and a cache flush (FlushGpuWrittenCommands: ~400 GPU round trips a
    // second in a Mario Kart World race, during which the back had nothing to do). Pages
    // FlushGpuWrittenCommands flushed whole and no GPU work wrote since are read from guest
    // memory.
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || count == 0 ||
        rasterizer == nullptr) {
        return false;
    }
    const GPUVAddr end = address + u64{count} * sizeof(u32);
    bool written = false;
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < end; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            Tegra::Nx2MarkCommandListPage(*device);
            written = true;
        }
    }
    if (!written) {
        return false;
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.WaitForWriters();
    }
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    command_headers.resize_destructive(count);
    u8* const out = reinterpret_cast<u8*>(command_headers.data());
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
        const std::span<u8> part{out + (chunk - address), static_cast<size_t>(chunk_end - chunk)};
        const auto device = memory_manager.GpuToCpuAddress(chunk);
        if (!device) {
            return false;
        }
        GpuWrittenList where = GpuWrittenList::Synced;
        if (Tegra::Nx2IsGpuWritten(*device)) {
            where = rasterizer->ReadGpuWrittenCommandList(*device, part);
        }
        if (where == GpuWrittenList::Unknown) {
            return false;
        }
        if (where == GpuWrittenList::Synced) {
            memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
        }
        chunk = chunk_end;
    }
    return true;
}

void DmaPusher::FlushGpuWrittenCommands(GPUVAddr address, size_t size) {
    // Title 1 runs small command lists its culling dispatches fill in each frame. Accurate GPU
    // mode reads every command list through the caches (safe reads); here only the ones on pages
    // the GPU has written are, or their draws run with an older frame's words (trees vanish).
    // These are the lists ReadGpuWrittenCommands can't take from a copy or guest memory (a page's
    // first list, or pages draws write too).
    if (!IsGpuWrittenCommands(address, size)) {
        return;
    }
    // The caches belong to the split's back: it first runs what the front queued (the dispatch
    // that writes these words among it). The whole pages are flushed: their next lists read
    // guest memory until GPU work writes them again.
    VideoCommon::GpuSplit::Get().Drain();
    const GPUVAddr first_page = address & ~GPUVAddr{0xfff};
    const GPUVAddr end_page = (address + size + 0xfff) & ~GPUVAddr{0xfff};
    if (memory_manager.IsMemoryDirty(first_page, end_page - first_page,
                                     VideoCommon::CacheType::BufferCache)) {
        // The download records into the Vulkan scheduler, which the split's back owns: a drained
        // back still submits from its idle hook, so the front must not flush these pages itself.
        VideoCommon::GpuSplit::Get().Run(
            [memory = &memory_manager, first_page, size = end_page - first_page] {
                memory->FlushRegion(first_page, size, VideoCommon::CacheType::BufferCache);
            });
    }
    if (rasterizer == nullptr) {
        return;
    }
    for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsCommandListPage(*device)) {
            rasterizer->NoteGpuWrittenCommandListSynced(*device);
        }
    }
}

void DmaPusher::ReportUnboundSubchannel() {
    // Methods sent to a subchannel with no engine (no SetObject yet) are dropped.
    static std::atomic<u32> reported{};
    if (reported.fetch_add(1) < 8) {
        LOG_WARNING(HW_GPU, "Dropping method {:#x} on unbound subchannel {}", dma_state.method,
                    dma_state.subchannel);
    }
}

void DmaPusher::CheckSubmitHash(const CommandListHeader& header, u64 submit_hash) {
    std::vector<u32> words(header.size);
    memory_manager.ReadBlockUnsafe(header.addr, words.data(), words.size() * sizeof(u32));
    static std::atomic<u64> checked{};
    static std::atomic<u64> changed{};
    ++checked;
    if (HashPushbufferWords(words) != submit_hash) {
        if (changed.fetch_add(1) < 16) {
            LOG_WARNING(HW_GPU, "pushbuffer entry {:#x}+{:#x} changed between submit and execution",
                        header.addr.Value(), header.size.Value());
        }
    }
    if ((checked % 20000) == 0) {
        LOG_WARNING(HW_GPU, "pushbuffer check: {} of {} entries changed after submit",
                    changed.load(), checked.load());
    }
}

void DmaPusher::RecordParsedEntry(const CommandListHeader& header,
                                  std::span<const CommandHeader> commands) {
    if (!g_record_parsed_entries) {
        return;
    }
    u64 hash = 0xcbf29ce484222325ULL;
    for (const CommandHeader& command : commands) {
        hash = (hash ^ command.argument) * 0x100000001b3ULL;
    }
    parsed_entries[parsed_entry_count++ % parsed_entries.size()] = {header.addr, static_cast<u32>(header.size), hash};
}

void DmaPusher::DumpCurrentEntry(const char* reason) {
    // Entries whose memory changed after they were parsed (the CPU rewrote them meanwhile).
    for (size_t i = 0; i < std::min(parsed_entry_count, parsed_entries.size()); ++i) {
        const auto& entry = parsed_entries[(parsed_entry_count - 1 - i) % parsed_entries.size()];
        std::vector<u32> words(entry.size);
        memory_manager.ReadBlockUnsafe(entry.address, words.data(), words.size() * sizeof(u32));
        u64 hash = 0xcbf29ce484222325ULL;
        for (const u32 word : words) {
            hash = (hash ^ word) * 0x100000001b3ULL;
        }
        LOG_WARNING(HW_GPU, "  parsed -{}: {:#x}+{:#x} {}", i, entry.address, entry.size,
                    hash == entry.hash ? "unchanged" : "CHANGED since parsed");
    }
    // EDEN_NX2_PB_DUMP=<file>: also write the whole command list (u64 address, u32 size and the
    // current words of every entry, then the current entry index) for offline replay.
    if (static const char* const path = std::getenv("EDEN_NX2_PB_DUMP");
        path != nullptr && !dma_pushbuffer.empty()) {
        if (std::FILE* file = std::fopen(path, "wb")) {
            const auto& lists = dma_pushbuffer.front().command_lists;
            const u64 current = dma_pushbuffer_subindex;
            const u64 count = lists.size();
            std::fwrite(&count, sizeof(count), 1, file);
            std::fwrite(&current, sizeof(current), 1, file);
            for (const CommandListHeader& header : lists) {
                const u64 address = header.addr;
                const u32 size = static_cast<u32>(header.size);
                std::vector<u32> words(size);
                memory_manager.ReadBlockUnsafe(address, words.data(), words.size() * sizeof(u32));
                std::fwrite(&address, sizeof(address), 1, file);
                std::fwrite(&size, sizeof(size), 1, file);
                std::fwrite(words.data(), sizeof(u32), words.size(), file);
            }
            std::fclose(file);
        }
    }
    LOG_WARNING(HW_GPU, "pushbuffer dump ({}): method={:#x} sub={} count={} non_inc={} get={:#x}",
                reason, dma_state.method, dma_state.subchannel, dma_state.method_count,
                dma_state.non_incrementing, dma_state.dma_get);
    if (dma_pushbuffer.empty()) {
        return;
    }
    const auto& lists = dma_pushbuffer.front().command_lists;
    const size_t first = dma_pushbuffer_subindex >= 2 ? dma_pushbuffer_subindex - 2 : 0;
    for (size_t i = first; i <= dma_pushbuffer_subindex && i < lists.size(); ++i) {
        const CommandListHeader& header = lists[i];
        LOG_WARNING(HW_GPU, "  entry {} of {}: addr={:#x} size={:#x}", i, lists.size(),
                    header.addr.Value(), header.size.Value());
        for (u64 word = 0; word < std::min<u64>(header.size, 0x80); word += 8) {
            std::string gpu_line;
            std::string cpu_line;
            for (u64 j = word; j < std::min<u64>(header.size, word + 8); ++j) {
                const GPUVAddr address = header.addr + j * sizeof(u32);
                const u32 gpu_value = memory_manager.Read<u32>(address);
                const u32 cpu_value = system.ApplicationMemory().Read32(address);
                gpu_line += fmt::format(" {:08x}", gpu_value);
                cpu_line += gpu_value == cpu_value ? std::string(" ........")
                                                   : fmt::format(" {:08x}", cpu_value);
            }
            LOG_WARNING(HW_GPU, "    +{:03x}:{}", word * 4, gpu_line);
            LOG_WARNING(HW_GPU, "    cpu :{}", cpu_line);
        }
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

} // namespace Tegra
