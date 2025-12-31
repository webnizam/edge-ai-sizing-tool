// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Workload } from '@/payload-types'

// Props for the ImageClassification component
export interface ImageClassificationProps {
  workload: Workload
}

// Performance metrics for FPS display (DLStreamer-style)
export interface ImageClassificationPerformanceMetrics {
  total_fps?: number | null
  average_fps_per_stream?: number | null
}
