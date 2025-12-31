# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Image Classification Worker for Edge AI Sizing Tool

This worker provides a FastAPI server for image classification using Intel DLStreamer.
It uses GStreamer pipelines with gvaclassify for real-time video classification
and displays labels on the video stream.
"""

import os
import re
import sys
import cv2
import time
import math
import socket
import signal
import logging
import uvicorn
import argparse
import threading
import urllib.parse
import numpy as np
import subprocess as sp
import requests

from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set environment variables to enable DLStreamer
os.environ["LIBVA_DRIVER_NAME"] = "iHD"
os.environ["GST_PLUGIN_PATH"] = (
    "/opt/intel/dlstreamer/lib:/opt/intel/dlstreamer/gstreamer/lib/gstreamer-1.0:/opt/intel/dlstreamer/streamer/lib/"
)
os.environ["LD_LIBRARY_PATH"] = (
    "/opt/intel/dlstreamer/gstreamer/lib:/opt/intel/dlstreamer/lib:/opt/intel/dlstreamer/lib/gstreamer-1.0:/sr/lib:/opt/intel/dlstreamer/lib:/usr/local/lib/gstreamer-1.0:/usr/local/lib:/opt/opencv:/opt/rdkafka"
)
os.environ["LIBVA_DRIVERS_PATH"] = "/usr/lib/x86_64-linux-gnu/dri"
os.environ["GST_VA_ALL_DRIVERS"] = "1"
os.environ["PATH"] = (
    f"/opt/intel/dlstreamer/gstreamer/bin:/opt/intel/dlstreamer/bin:{os.environ['PATH']}"
)
os.environ["GST_PLUGIN_FEATURE_RANK"] = (
    os.environ.get("GST_PLUGIN_FEATURE_RANK", "") + ",ximagesink:MAX"
)
os.environ["GI_TYPELIB_PATH"] = (
    "/opt/intel/dlstreamer/gstreamer/lib/girepository-1.0:/usr/lib/x86_64-linux-gnu/girepository-1.0"
)

env = os.environ.copy()
venv_path = os.path.dirname(sys.executable)
env["PATH"] = f"{venv_path}:{env['PATH']}"

VIDEO_DIR = Path("../assets/media")
MODEL_DIR = Path("./models")
CUSTOM_MODELS_DIR = Path("../custom_models/image-classification")


def is_valid_id(workload_id):
    """Validate that workload_id is a positive integer."""
    return isinstance(workload_id, int) and workload_id > 0


def update_payload_status(workload_id: int, status):
    """
    Update the workload status in a safe way, allow-listing scheme, authority,
    and preventing unsafe path traversal.
    """
    if not is_valid_id(workload_id):
        logging.error(f"Invalid workload ID: {workload_id}. Refusing to update status.")
        return

    allowed_scheme = "http"
    allowed_netloc = "127.0.0.1:8080"
    path = f"/api/workloads/{workload_id}"
    composed_url = f"{allowed_scheme}://{allowed_netloc}{path}"
    parsed_url = urllib.parse.urlparse(composed_url)

    if parsed_url.scheme != allowed_scheme or parsed_url.netloc != allowed_netloc:
        logging.error(f"URL scheme or authority not allowed: {parsed_url.geturl()}")
        return

    if ".." in path or "//" in path or " " in path:
        logging.error(f"Invalid characters in URL path: {path}")
        return

    url = parsed_url.geturl()
    data = {"status": status, "port": args.port}
    try:
        response = requests.patch(url, json=data)
        response.raise_for_status()
        logging.info(f"Successfully updated status to {status} for {workload_id}.")
    except requests.exceptions.RequestException as e:
        logging.info(f"Failed to update status: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("--- Initializing image classification worker ---")
    app.state.pipeline_metrics = {
        "total_fps": None,
        "number_streams": None,
        "average_fps_per_stream": None,
        "fps_streams": None,
        "timestamp": None,
    }
    thread = threading.Thread(target=main, daemon=True)
    thread.start()
    yield
    logging.info("--- Shutting down image classification worker ---")
    thread.join()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FastAPI server for Intel® DLStreamer image classification"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=f"{VIDEO_DIR}/people-detection.mp4",
        help="Input source e.g. /dev/video0, videofile.mp4, etc",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet-18-pytorch",
        help="Model name (default: resnet-18-pytorch)",
    )
    parser.add_argument(
        "--model_parent_dir",
        type=str,
        default=MODEL_DIR,
        help=f"Path to the model directory (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--model_precision",
        type=str,
        default="FP16",
        help="Model precision (default: FP16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Device to run inference on (default: CPU)",
    )
    parser.add_argument(
        "--decode_device",
        type=str,
        default="CPU",
        help="Device to run decode on (default: CPU)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--tcp_port",
        type=int,
        default=5000,
        help="Port to spawn the DLStreamer pipeline (default: 5000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5024,
        help="Port to run the FastAPI server on (default: 5024)",
    )
    parser.add_argument(
        "--id",
        type=int,
        help="Workload ID to update the workload status",
    )
    parser.add_argument(
        "--number_of_streams",
        type=int,
        default=1,
        help="Number of streams to run (default: 1)",
    )
    parser.add_argument(
        "--width_limit",
        type=int,
        default=640,
        help="Width limit for the video stream (default: 640)",
    )
    parser.add_argument(
        "--height_limit",
        type=int,
        default=480,
        help="Height limit for the video stream (default: 480)",
    )
    return parser.parse_args()


args = parse_arguments()


def get_model_path(model_name: str, precision: str) -> tuple[Path | None, Path | None, Path | None]:
    """
    Get the path to the model file, labels file, and model-proc file.
    Returns (model_path, labels_path, model_proc_path) tuple.
    """
    # Check standard model directory first
    model_dir = Path(args.model_parent_dir) / model_name / precision
    model_path = model_dir / f"{model_name}.xml"
    labels_path = model_dir / "labels.txt"
    model_proc_path = model_dir / "model-proc.json"
    
    if model_path.exists():
        logging.info(f"Found model at {model_path}")
        if not labels_path.exists():
            # Check for labels in parent directory
            parent_labels = Path(args.model_parent_dir) / model_name / "labels.txt"
            if parent_labels.exists():
                labels_path = parent_labels
            else:
                labels_path = None
        if not model_proc_path.exists():
            # Check for model-proc in parent directory
            parent_model_proc = Path(args.model_parent_dir) / model_name / "model-proc.json"
            if parent_model_proc.exists():
                model_proc_path = parent_model_proc
            else:
                model_proc_path = None
        return model_path, labels_path, model_proc_path
    
    # Check custom models directory
    custom_model_dir = CUSTOM_MODELS_DIR / model_name
    if custom_model_dir.exists():
        xml_files = list(custom_model_dir.glob("*.xml"))
        if xml_files:
            model_path = xml_files[0]
            logging.info(f"Found custom model at {model_path}")
            custom_labels_path = custom_model_dir / "labels.txt"
            if custom_labels_path.exists():
                labels_path = custom_labels_path
            else:
                labels_path = None
            custom_model_proc = custom_model_dir / "model-proc.json"
            if custom_model_proc.exists():
                model_proc_path = custom_model_proc
            else:
                model_proc_path = None
            return model_path, labels_path, model_proc_path
    
    logging.warning(f"Model not found at {model_dir} or {custom_model_dir}")
    return None, None, None


def build_compositor_props(num_streams, final_width, final_height):
    """
    Build compositor properties for grid layout of multiple streams.
    """
    grid_cols = math.ceil(math.sqrt(num_streams))
    grid_rows = math.ceil(num_streams / grid_cols)
    sub_width = final_width // grid_cols
    sub_height = final_height // grid_rows

    comp_props = []
    for i in range(num_streams):
        row = i // grid_cols
        col = i % grid_cols
        xpos = col * sub_width
        ypos = row * sub_height
        comp_props.append(
            f"sink_{i}::xpos={xpos} "
            f"sink_{i}::ypos={ypos} "
            f"sink_{i}::width={sub_width} "
            f"sink_{i}::height={sub_height}"
        )
    return " ".join(comp_props)


def is_webcam_input(input_source: str) -> bool:
    """Check if the input is a webcam device."""
    return (
        input_source.startswith("/dev/video") or 
        input_source.isdigit() or 
        (not input_source.endswith((".mp4", ".avi", ".mov")) and not input_source.startswith("rtsp://"))
    )


def build_pipeline(
    tcp_port,
    input,
    model_full_path,
    model_label_path,
    model_proc_path,
    device,
    decode_device,
    batch_size=1,
    number_of_streams=1,
):
    """
    Build the DLStreamer pipeline for image classification with MJPEG streaming.
    Uses gvaclassify for classification inference.
    """
    
    # Determine if this is a webcam input
    webcam_input = is_webcam_input(input)

    # Check if input is a videofile
    if input.endswith((".mp4", ".avi", ".mov")):
        source_command = ["multifilesrc", f"location={input}", "loop=true"]
    elif input.startswith("rtsp://"):
        source_command = ["rtspsrc", f"location={input}", "protocols=tcp"]
    else:
        # Default to webcam - ensure proper device path
        device_path = input if input.startswith("/dev/video") else f"/dev/video{input}"
        if number_of_streams > 1:
            source_command = [
                "v4l2src",
                f"device={device_path}",
                "!",
                "videoconvert",
                "!",
                "tee",
                "name=camtee",
                "!",
                "multiqueue",
                "name=camq",
            ]
        else:
            source_command = ["v4l2src", f"device={device_path}"]

    # Configure decode element based on input type
    if webcam_input:
        # For webcam: v4l2src outputs raw video, just need videoconvert
        decode_element = ["videoconvert", "!", "video/x-raw"]
    elif input.endswith((".mp4", ".avi", ".mov")):
        # For video files: use decodebin3
        decode_element = ["decodebin3", "!", "videoconvert", "!", "video/x-raw"]
    elif input.startswith("rtsp://"):
        # For RTSP streams
        if "GPU" in decode_device:
            decode_element = [
                "rtph264depay",
                "!",
                "avdec_h264",
                "!",
                "vapostproc",
                "!",
                "video/x-raw(memory:VAMemory)",
            ]
        else:
            decode_element = [
                "rtph264depay",
                "!",
                "avdec_h264",
                "!",
                "videoconvert",
                "!",
                "video/x-raw",
            ]
    else:
        # Fallback to decodebin3
        decode_element = ["decodebin3", "!", "videoconvert", "!", "video/x-raw"]

    # Get the directory where this script is located for gvapython callback
    script_dir = Path(__file__).parent.resolve()
    draw_script = script_dir / "draw_classification.py"

    # Configure inference command for classification using gvaclassify
    inference_command = [
        "gvaclassify",
        f"model={model_full_path}",
        f"device={device}",
        "inference-region=full-frame",  # Enable full-frame classification
    ]

    if model_label_path is not None:
        inference_command.append(f"labels-file={model_label_path}")
    
    if model_proc_path is not None:
        inference_command.append(f"model-proc={model_proc_path}")

    if "GPU" in decode_device and "GPU" in device:
        inference_command.append(f"batch-size={batch_size}")
        inference_command.append("nireq=4")
        inference_command.append("pre-process-backend=va-surface-sharing")
    elif "GPU" in decode_device and "CPU" in device:
        inference_command.append("pre-process-backend=va")

    comp_props_str = build_compositor_props(
        args.number_of_streams, args.width_limit, args.height_limit
    )
    comp_props = comp_props_str.split()
    logging.info(f"Compositor properties: {comp_props_str}")

    # Build the compositor pipeline
    pipeline = (
        ["gst-launch-1.0", "compositor", "name=comp"]
        + comp_props
        + ["!"]
        + ["jpegenc", "!", "multipartmux", "boundary=frame"]
        + ["!"]
        + ["tcpserversink", f"host=127.0.0.1", f"port={tcp_port}"]
    )
    logging.info(f"Partial Pipeline={pipeline}")

    # Compose the full pipeline
    if input.startswith("/dev/video") and number_of_streams > 1:
        # For multiple webcam streams, use tee to split the source
        pipeline += source_command
        for i in range(number_of_streams):
            pipeline += (
                [
                    f"camtee.",
                    "!",
                    "queue",
                    "max-size-buffers=10",
                    "leaky=downstream",
                    "!",
                ]
                + decode_element
                + [
                    "!",
                    *inference_command,
                    "!",
                    "queue",
                    "!",
                    "gvafpscounter",
                    "!",
                    "gvawatermark",
                    "!",
                    "videoconvert",
                    "!",
                    f"comp.sink_{i}",
                ]
            )
    else:
        for i in range(number_of_streams):
            pipeline += (
                source_command
                + ["!"]
                + decode_element
                + ["!"]
                + inference_command
                + [
                    "!",
                    "queue",
                    "!",
                    "gvafpscounter",
                    "!",
                    "gvawatermark",
                    "!",
                    "videoconvert",
                    "!",
                    f"comp.sink_{i}",
                ]
            )
    
    logging.info(f"Full pipeline={' '.join(pipeline)}\n")
    return pipeline


def stop_signal_handler(sig, frame):
    """Signal handler for SIGINT to terminate worker"""
    logging.info("SIGINT received. Stopping the application...")
    exit(0)


signal.signal(signal.SIGINT, stop_signal_handler)


def run_pipeline(pipeline):
    """
    Run the GStreamer pipeline and process its output in real-time.
    """
    logging.info("Starting GStreamer classification pipeline...")
    try:
        process = sp.Popen(pipeline, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        for line in process.stdout:
            logging.info(line.strip())
            metrics = filter_result(line.strip())
            if metrics:
                app.state.pipeline_metrics.update(metrics)

        for error_line in process.stderr:
            logging.error(error_line.strip())

        if process.returncode == 0 or process.returncode is None:
            logging.info("Pipeline reached EOS. Restarting...")
            process.communicate()
        else:
            logging.error(f"Pipeline exited with error code: {process.returncode}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if process and process.poll() is None:
            process.terminate()
            process.wait()
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait()


def filter_result(output):
    """
    Extract the FPS metrics from the command output.
    """
    fps_pattern = re.compile(
        r"FpsCounter\(.*\): total=(\d+\.\d+) fps, number-streams=(\d+), per-stream=(\d+\.\d+) fps(?: \((.*?)\))?"
    )
    match = fps_pattern.search(output)
    if match:
        total_fps_str = match.group(1)
        number_streams_str = match.group(2)
        average_fps_per_stream_str = match.group(3)
        all_streams_fps_str = match.group(4)

        total_fps = float(total_fps_str)
        number_streams = int(number_streams_str)
        average_fps_per_stream = float(average_fps_per_stream_str)

        if all_streams_fps_str:
            all_streams_fps = [float(x.strip()) for x in all_streams_fps_str.split(",")]
        else:
            all_streams_fps = [average_fps_per_stream]

        fps_streams = {
            f"stream_id {i+1}": fps_val
            for i, fps_val in enumerate(all_streams_fps[:number_streams])
        }

        return {
            "total_fps": total_fps,
            "number_streams": number_streams,
            "average_fps_per_stream": average_fps_per_stream,
            "fps_streams": fps_streams,
            "timestamp": time.time(),
        }
    return None


def main():
    """
    Main function to initialize and run the classification pipeline.
    """
    model_path, labels_path, model_proc_path = get_model_path(args.model, args.model_precision)
    
    if model_path is None:
        logging.error("Model not found. Exiting.")
        if args.id:
            update_payload_status(args.id, "error")
        return

    logging.info(f"Using model: {model_path}")
    if labels_path:
        logging.info(f"Using labels: {labels_path}")
    if model_proc_path:
        logging.info(f"Using model-proc: {model_proc_path}")

    if args.id:
        update_payload_status(args.id, "active")

    while True:
        pipeline = build_pipeline(
            tcp_port=args.tcp_port,
            input=args.input,
            model_full_path=str(model_path),
            model_label_path=str(labels_path) if labels_path else None,
            model_proc_path=str(model_proc_path) if model_proc_path else None,
            device=args.device,
            decode_device=args.decode_device,
            batch_size=args.batch_size,
            number_of_streams=args.number_of_streams,
        )
        run_pipeline(pipeline)
        logging.info("Pipeline finished. Restarting in 5 seconds...")
        time.sleep(5)


def mjpeg_stream(host: str = "127.0.0.1", port: int = 5000):
    """
    Connect to the GStreamer TCP server and yield MJPEG frames.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        buffer = b""

        while True:
            data = client_socket.recv(4096)
            if not data:
                break

            buffer += data
            while b"\r\n\r\n" in buffer:
                frame, _, buffer = buffer.partition(b"\r\n\r\n")

                try:
                    np_frame = np.frombuffer(frame, dtype=np.uint8)
                    image = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
                    if image is None:
                        continue

                    _, jpeg_frame = cv2.imencode(".jpg", image)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg_frame.tobytes()
                        + b"\r\n"
                    )
                except Exception as e:
                    logging.info(f"Error processing frame: {e}")


@app.get("/result")
def get_mjpeg_stream():
    """
    Serve the MJPEG stream as an HTTP response.
    """
    try:
        return StreamingResponse(
            mjpeg_stream(port=args.tcp_port),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        return JSONResponse(
            {
                "status": False,
                "message": "An error occurred while retrieving mjpeg stream",
            }
        )


@app.get("/api/metrics")
def get_pipeline_metrics():
    """
    Return the current pipeline metrics.
    """
    try:
        result = {
            "data": app.state.pipeline_metrics,
            "status": "success",
        }
        return JSONResponse(result)
    except Exception as e:
        logging.error(f"Error retrieving metrics: {e}")
        return JSONResponse(
            {"status": False, "message": "An error occurred while retrieving metrics"}
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=args.port)
