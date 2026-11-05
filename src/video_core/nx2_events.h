// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace VideoCommon::Nx2Events {

/// EDEN_NX2_EVENTS=1: counts named events (render pass ends, queue submissions, how conditional
/// rendering was resolved) and logs the counts every 10 s; for events counted with their stacks,
/// also the most common call stacks, with function names.
[[nodiscard]] bool Enabled();

/// Counts an event. The name must be a string literal (it is kept, not copied).
void Count(const char* name);

/// Counts an event and the call stack it came from.
void CountStack(const char* name);

} // namespace VideoCommon::Nx2Events
