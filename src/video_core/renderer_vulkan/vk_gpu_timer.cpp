// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <utility>

#include "common/logging.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_gpu_timer.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

bool GpuTimer::Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_TIMESTAMPS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

GpuTimer::GpuTimer(const Device& device_) : device{device_} {
    const float period = device.GetTimestampPeriod();
    if (period <= 0.0f || !device.IsHostQueryResetSupported()) {
        LOG_WARNING(Render_Vulkan, "GPU timestamps unavailable (period {}, host query reset {})",
                    period, device.IsHostQueryResetSupported());
        return;
    }
    ns_per_tick = period;
    pool = device.GetLogical().CreateQueryPool({
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = MaxRegions * 2,
        .pipelineStatistics = 0,
    });
    device.GetLogical().ResetQueryPool(*pool, 0, MaxRegions * 2);
    regions.resize(MaxRegions);
    results.resize(size_t{MaxRegions} * 4);
    LOG_INFO(Render_Vulkan, "GPU timestamps on ({} ns per tick)", ns_per_tick);
}

GpuTimer::~GpuTimer() = default;

u64 GpuTimer::RenderPassLabel(u32 width, u32 height, u32 images, bool depth,
                              VkRenderPass renderpass) {
    // The render pass object stands for the attachment formats; 19 bits of a hash of it tell
    // passes of the same size apart.
    const u64 tag = (reinterpret_cast<u64>(renderpass) * 0x9E3779B97F4A7C15ULL) >> 45;
    const u64 key = u64{width & 0xffff} | (u64{height & 0xffff} << 16) | (u64{images & 0xf} << 32) |
                    (u64{depth ? 1U : 0U} << 36) | (tag << 37);
    return Label(Kind::RenderPass, key);
}

u32 GpuTimer::Begin(vk::CommandBuffer cmdbuf, u64 label) {
    if (!pool) {
        return NoRegion;
    }
    if (in_flight >= MaxRegions) {
        ++dropped;
        return NoRegion;
    }
    const u32 region = next;
    next = (next + 1) % MaxRegions;
    ++in_flight;
    ++batch_count;
    regions[region] = {label, false};
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *pool, region * 2);
    return region;
}

void GpuTimer::End(vk::CommandBuffer cmdbuf, u32 region) {
    if (region == NoRegion) {
        return;
    }
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pool, region * 2 + 1);
    regions[region].ended = true;
}

void GpuTimer::Submitted(u64 tick) {
    if (!pool) {
        return;
    }
    ++submissions;
    if (batch_count == 0) {
        return;
    }
    batches.push_back({tick, batch_first, batch_count});
    batch_first = (batch_first + batch_count) % MaxRegions;
    batch_count = 0;
}

void GpuTimer::Collect(const MasterSemaphore& semaphore) {
    if (!pool) {
        return;
    }
    while (!batches.empty() && semaphore.IsFree(batches.front().tick)) {
        ReadBatch(batches.front());
        in_flight -= batches.front().count;
        batches.pop_front();
    }
    const auto now = std::chrono::steady_clock::now();
    if (since == std::chrono::steady_clock::time_point{}) {
        since = now;
    }
    const double seconds = std::chrono::duration<double>(now - since).count();
    if (seconds * 1e9 >= static_cast<double>(VideoCommon::Nx2Profile::ReportWindowNs())) {
        Report(seconds);
        totals.clear();
        submissions = 0;
        dropped = 0;
        since = now;
    }
}

void GpuTimer::ReadBatch(const Batch& batch) {
    const u32 first_part = (std::min)(batch.count, MaxRegions - batch.first);
    ReadRange(batch.first, first_part);
    if (first_part < batch.count) {
        ReadRange(0, batch.count - first_part);
    }
}

void GpuTimer::ReadRange(u32 first, u32 count) {
    const u32 queries = count * 2;
    // Each query gives its value and whether it is available.
    const VkResult result = device.GetLogical().GetQueryResults(
        *pool, first * 2, queries, size_t{queries} * 2 * sizeof(u64), results.data(),
        2 * sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result == VK_SUCCESS || result == VK_NOT_READY) {
        for (u32 i = 0; i < count; ++i) {
            const Region& region = regions[first + i];
            const u64 begin = results[size_t{i} * 4];
            const bool begin_ready = results[size_t{i} * 4 + 1] != 0;
            const u64 end = results[size_t{i} * 4 + 2];
            const bool end_ready = results[size_t{i} * 4 + 3] != 0;
            if (!region.ended || !begin_ready || !end_ready || end < begin) {
                continue;
            }
            Total& total = totals[region.label];
            total.ns += static_cast<u64>(static_cast<double>(end - begin) * ns_per_tick);
            ++total.count;
        }
    }
    device.GetLogical().ResetQueryPool(*pool, first * 2, queries);
}

void GpuTimer::Report(double seconds) {
    const auto ms = [](u64 ns) { return static_cast<double>(ns) / 1e6; };
    u64 command_buffer_ns = 0;
    u64 command_buffers = 0;
    u64 pass_ns = 0;
    u64 passes = 0;
    u64 dispatch_ns = 0;
    u64 dispatches = 0;
    std::vector<std::pair<u64, u64>> ranked; // (ns, label)
    for (const auto& [label, total] : totals) {
        switch (static_cast<Kind>(label >> 56)) {
        case Kind::CommandBuffer:
            command_buffer_ns += total.ns;
            command_buffers += total.count;
            break;
        case Kind::RenderPass:
            pass_ns += total.ns;
            passes += total.count;
            ranked.emplace_back(total.ns, label);
            break;
        case Kind::Dispatch:
            dispatch_ns += total.ns;
            dispatches += total.count;
            ranked.emplace_back(total.ns, label);
            break;
        }
    }
    const u64 other_ns =
        command_buffer_ns > pass_ns + dispatch_ns ? command_buffer_ns - pass_ns - dispatch_ns : 0;
    LOG_INFO(Render_Vulkan,
             "GPU time in the last {:.1f} s: {:.1f} ms in {} command buffers ({:.0f}% of the "
             "time), render passes {:.1f} ms ({}), dispatches {:.1f} ms ({}), the rest {:.1f} ms; "
             "{} submissions, {} regions not timed",
             seconds, ms(command_buffer_ns), command_buffers,
             100.0 * ms(command_buffer_ns) / (seconds * 1000.0), ms(pass_ns), passes,
             ms(dispatch_ns), dispatches, ms(other_ns), submissions, dropped);
    std::ranges::sort(ranked, std::greater{});
    for (size_t i = 0; i < (std::min<size_t>)(ranked.size(), 20); ++i) {
        const auto [ns, label] = ranked[i];
        const u64 count = totals[label].count;
        const u64 key = label & ((u64{1} << 56) - 1);
        const double each = ms(ns) / static_cast<double>((std::max<u64>)(count, 1));
        if (static_cast<Kind>(label >> 56) == Kind::RenderPass) {
            LOG_INFO(Render_Vulkan,
                     "  {:7.1f} ms {:6} x {:7.3f} ms  render pass {}x{}, {} images{}, kind {:05x}",
                     ms(ns), count, each, key & 0xffff, (key >> 16) & 0xffff, (key >> 32) & 0xf,
                     ((key >> 36) & 1) != 0 ? " with depth" : "", key >> 37);
        } else {
            LOG_INFO(Render_Vulkan, "  {:7.1f} ms {:6} x {:7.3f} ms  dispatch of shader {:014x}",
                     ms(ns), count, each, key);
        }
    }
}

} // namespace Vulkan
