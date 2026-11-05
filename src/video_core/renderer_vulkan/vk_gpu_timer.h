// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <deque>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MasterSemaphore;

/// EDEN_NX2_GPU_TIMESTAMPS=1: times command buffers, render passes and compute dispatches on the
/// GPU with timestamp queries, and every 10 s logs where the GPU's time went. Only the scheduler's
/// worker thread calls it (and the scheduler's constructor, before the worker starts).
class GpuTimer {
public:
    enum class Kind : u8 {
        CommandBuffer = 1,
        RenderPass = 2,
        Dispatch = 3,
    };
    static constexpr u32 NoRegion = ~0U;

    explicit GpuTimer(const Device& device);
    ~GpuTimer();

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    /// Whether EDEN_NX2_GPU_TIMESTAMPS asks for it.
    [[nodiscard]] static bool Enabled();

    [[nodiscard]] static u64 Label(Kind kind, u64 key) {
        return (u64{static_cast<u8>(kind)} << 56) | (key & ((u64{1} << 56) - 1));
    }

    /// A render pass's label: its size, attachments and render pass object.
    [[nodiscard]] static u64 RenderPassLabel(u32 width, u32 height, u32 images, bool depth,
                                             VkRenderPass renderpass);

    /// Starts timing a region of `cmdbuf`; NoRegion when timing is off or no query is free.
    u32 Begin(vk::CommandBuffer cmdbuf, u64 label);
    void End(vk::CommandBuffer cmdbuf, u32 region);

    /// The regions begun since the last call belong to the submission that signals `tick`.
    void Submitted(u64 tick);

    /// Reads the regions of finished submissions; logs every 10 s.
    void Collect(const MasterSemaphore& semaphore);

private:
    struct Region {
        u64 label;
        bool ended;
    };
    struct Batch {
        u64 tick;
        u32 first;
        u32 count;
    };
    struct Total {
        u64 ns;
        u64 count;
    };

    void ReadBatch(const Batch& batch);
    void ReadRange(u32 first, u32 count);
    void Report(double seconds);

    static constexpr u32 MaxRegions = 16384; // two queries each

    const Device& device;
    vk::QueryPool pool;
    double ns_per_tick = 0.0;
    std::vector<Region> regions;
    u32 next = 0;
    u32 in_flight = 0;
    u32 batch_first = 0;
    u32 batch_count = 0;
    std::deque<Batch> batches;
    std::vector<u64> results;
    std::unordered_map<u64, Total> totals;
    u64 submissions = 0;
    u64 dropped = 0;
    std::chrono::steady_clock::time_point since{};
};

} // namespace Vulkan
