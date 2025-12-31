# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GVA Python callback for drawing classification labels on video frames.
This script is used with gvapython element to render classification results.
"""

import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo
import numpy as np

# Try to import cv2 for drawing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Global variables to store the latest classification result
last_label = "Processing..."
last_confidence = 0.0


def process_frame(frame):
    """
    Process each frame and draw classification label.
    This function is called by gvapython for every frame.
    """
    global last_label, last_confidence
    
    try:
        # Get classification results from frame tensors
        for tensor in frame.tensors():
            if tensor.has_field("label"):
                last_label = tensor.label()
            if tensor.has_field("confidence"):
                last_confidence = tensor.confidence()
            # Also try name field
            if tensor.has_field("name"):
                last_label = tensor["name"]
            if tensor.has_field("label_id"):
                label_id = tensor["label_id"]
                # Map label_id to label if needed
        
        # Check for classification results in messages
        for message in frame.messages():
            try:
                if hasattr(message, 'label'):
                    last_label = message.label()
                if hasattr(message, 'confidence'):
                    last_confidence = message.confidence()
            except:
                pass
        
        # Also check regions (in case there's a full-frame region)
        for region in frame.regions():
            for tensor in region.tensors():
                if tensor.has_field("label"):
                    last_label = tensor.label()
                if tensor.has_field("confidence"):
                    last_confidence = tensor.confidence()
        
        if HAS_CV2:
            # Get frame data and draw on it
            with frame.data() as mat:
                if mat is not None:
                    height, width = mat.shape[:2]
                    
                    # Create label text with confidence
                    if last_confidence > 0:
                        label_text = f"{last_label}: {last_confidence:.1%}"
                    else:
                        label_text = last_label
                        
                    # Font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    font_thickness = 2
                    
                    # Calculate text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, font, font_scale, font_thickness
                    )
                    
                    # Position: top-left corner with padding
                    padding = 15
                    x = padding
                    y = padding + text_height + baseline
                    
                    # Draw background rectangle (semi-transparent black)
                    cv2.rectangle(
                        mat,
                        (x - 10, y - text_height - baseline - 10),
                        (x + text_width + 10, y + 10),
                        (0, 0, 0),  # Black background
                        -1  # Filled
                    )
                    
                    # Choose text color based on label
                    label_lower = last_label.lower() if last_label else ""
                    if label_lower in ["nsfw", "unsafe", "explicit", "1"]:
                        text_color = (0, 0, 255)  # Red for NSFW (BGR)
                    elif label_lower in ["normal", "safe", "clean", "sfw", "0"]:
                        text_color = (0, 255, 0)  # Green for normal (BGR)
                    else:
                        text_color = (255, 255, 255)  # White for unknown
                    
                    # Draw text
                    cv2.putText(
                        mat,
                        label_text,
                        (x, y),
                        font,
                        font_scale,
                        text_color,
                        font_thickness,
                        cv2.LINE_AA
                    )
    except Exception as e:
        print(f"Error in process_frame: {e}", file=sys.stderr)
    
    return True

