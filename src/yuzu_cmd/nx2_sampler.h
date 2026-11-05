// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Nx2Sampler {

// EDEN_NX2_SAMPLE=<thread names>: a sampling profiler for named host threads (Windows only).
// Names are comma separated; a name ending in '*' matches by prefix (CPUCore_*). From
// EDEN_NX2_SAMPLE_START seconds (default 60) for EDEN_NX2_SAMPLE_SECONDS (default 30), each
// thread is paused about 1000 times a second and its call stack recorded; the counts per
// function (where the thread is, and everything on its stack) go to EDEN_NX2_SAMPLE_OUT
// (default nx2_samples.txt).
// EDEN_NX2_SAMPLE_TRIGGER=<file>: sample on demand instead, whenever that file appears (it holds
// the seconds, default 20); the report goes to <file>-<n>.txt. Without EDEN_NX2_SAMPLE, the
// emulator's main threads are sampled.
void StartFromEnvironment();

} // namespace Nx2Sampler
