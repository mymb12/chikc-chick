import asyncio
import concurrent.futures
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

HOST = "0.0.0.0"
PORT = 6565
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 70
FRAME_DELAY = 1 / 15
CAMERA_INIT_TIMEOUT = 10

STATIC_FILES_DIR = Path(".")
STATIC_VIDEO_PATH = STATIC_FILES_DIR / "static" / "chicken_demo.mp4"
STATIC_VIDEO_INIT_TIMEOUT = 5

MOTION_BLUR_KERNEL = (21, 21)
MOTION_THRESHOLD = 25
MOTION_DILATE_ITERATIONS = 2
MOTION_MIN_AREA = 500

CLIENTS = set()
SUBSCRIPTIONS = {"normal_video": set(), "static_video": set()}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

CAMERA = None
camera_lock = asyncio.Lock()
CAMERA_TASK = None

STATIC_VIDEO_CAPTURE = None
static_video_lock = asyncio.Lock()
STATIC_VIDEO_TASK = None

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _initialize_camera_blocking():
    logging.info(f"[Executor] Attempting cv2.VideoCapture({WEBCAM_INDEX})...")
    try:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap or not cap.isOpened():
            logging.error(
                f"[Executor] cv2.VideoCapture({WEBCAM_INDEX}) failed to open."
            )
            return None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(
                f"[Executor] Webcam {WEBCAM_INDEX} opened. Resolution: {w}x{h}."
            )
            return cap
    except Exception as e:
        logging.error(
            f"[Executor] Exception during cv2.VideoCapture: {e}", exc_info=True
        )
        return None


async def initialize_camera():
    global CAMERA
    async with camera_lock:
        if CAMERA is not None and CAMERA.isOpened():
            return True
        logging.info(
            f"Requesting camera initialization via executor (timeout: {CAMERA_INIT_TIMEOUT}s)..."
        )
        loop = asyncio.get_running_loop()
        try:
            cap = await asyncio.wait_for(
                loop.run_in_executor(executor, _initialize_camera_blocking),
                timeout=CAMERA_INIT_TIMEOUT,
            )
            if cap is not None:
                logging.info("Camera initialization successful.")
                CAMERA = cap
                return True
            else:
                logging.error("Camera initialization failed (returned None).")
                CAMERA = None
                return False
        except asyncio.TimeoutError:
            logging.error(f"Camera initialization timed out.")
            CAMERA = None
            return False
        except Exception as e:
            logging.error(
                f"Unexpected error during camera initialization: {e}", exc_info=True
            )
            CAMERA = None
            return False


def _release_camera_blocking(cam_obj):
    if cam_obj:
        logging.info("[Executor] Releasing camera object...")
        try:
            cam_obj.release()
            logging.info("[Executor] Camera object released.")
        except Exception as e:
            logging.error(
                f"[Executor] Exception during camera.release(): {e}", exc_info=True
            )


async def release_camera():
    global CAMERA, CAMERA_TASK
    async with camera_lock:
        cam_to_release, task_to_cancel = CAMERA, CAMERA_TASK
        if cam_to_release is None and (task_to_cancel is None or task_to_cancel.done()):
            return
        logging.info("Initiating camera release process...")
        CAMERA, CAMERA_TASK = None, None
        if task_to_cancel and not task_to_cancel.done():
            logging.info("Cancelling camera broadcast task...")
            task_to_cancel.cancel()
            try:
                await task_to_cancel
            except asyncio.CancelledError:
                logging.info("Camera task cancelled.")
            except Exception as e:
                logging.error(f"Error awaiting cancelled camera task: {e}")
        if cam_to_release is not None:
            logging.info("Requesting camera release via executor...")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    executor, _release_camera_blocking, cam_to_release
                )
            except Exception as e:
                logging.error(f"Error running release in executor: {e}")
        logging.info("Camera release process finished.")


def _initialize_static_video_blocking():
    logging.info(f"[Executor] Attempting cv2.VideoCapture({STATIC_VIDEO_PATH})...")
    if not STATIC_VIDEO_PATH.is_file():
        logging.error(f"[Executor] Static video file not found: {STATIC_VIDEO_PATH}")
        return None
    try:
        cap = cv2.VideoCapture(str(STATIC_VIDEO_PATH))
        if not cap or not cap.isOpened():
            logging.error(
                f"[Executor] cv2.VideoCapture({STATIC_VIDEO_PATH}) failed to open."
            )
            return None
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logging.info(
                f"[Executor] Static video '{STATIC_VIDEO_PATH.name}' opened. Resolution: {w}x{h}, FPS: {fps:.2f}."
            )
            return cap
    except Exception as e:
        logging.error(
            f"[Executor] Exception during static video VideoCapture: {e}", exc_info=True
        )
        return None


async def initialize_static_video():
    global STATIC_VIDEO_CAPTURE
    async with static_video_lock:
        if STATIC_VIDEO_CAPTURE is not None and STATIC_VIDEO_CAPTURE.isOpened():
            return True
        logging.info(
            f"Requesting static video initialization via executor (timeout: {STATIC_VIDEO_INIT_TIMEOUT}s)..."
        )
        loop = asyncio.get_running_loop()
        try:
            cap = await asyncio.wait_for(
                loop.run_in_executor(executor, _initialize_static_video_blocking),
                timeout=STATIC_VIDEO_INIT_TIMEOUT,
            )
            if cap is not None:
                logging.info("Static video initialization successful.")
                STATIC_VIDEO_CAPTURE = cap
                return True
            else:
                logging.error("Static video initialization failed (returned None).")
                STATIC_VIDEO_CAPTURE = None
                return False
        except asyncio.TimeoutError:
            logging.error(f"Static video initialization timed out.")
            STATIC_VIDEO_CAPTURE = None
            return False
        except Exception as e:
            logging.error(
                f"Unexpected error during static video initialization: {e}",
                exc_info=True,
            )
            STATIC_VIDEO_CAPTURE = None
            return False


def _release_static_video_blocking(cap_obj):
    if cap_obj:
        logging.info("[Executor] Releasing static video capture object...")
        try:
            cap_obj.release()
            logging.info("[Executor] Static video capture object released.")
        except Exception as e:
            logging.error(
                f"[Executor] Exception during static video release: {e}", exc_info=True
            )


async def release_static_video():
    global STATIC_VIDEO_CAPTURE, STATIC_VIDEO_TASK
    async with static_video_lock:
        cap_to_release, task_to_cancel = STATIC_VIDEO_CAPTURE, STATIC_VIDEO_TASK
        if cap_to_release is None and (task_to_cancel is None or task_to_cancel.done()):
            return
        logging.info("Initiating static video release process...")
        STATIC_VIDEO_CAPTURE, STATIC_VIDEO_TASK = None, None  # Reset global state first
        if task_to_cancel and not task_to_cancel.done():
            logging.info("Cancelling static video broadcast task...")
            task_to_cancel.cancel()
            try:
                await task_to_cancel
            except asyncio.CancelledError:
                logging.info("Static video task cancelled.")
            except Exception as e:
                logging.error(f"Error awaiting cancelled static video task: {e}")
        if cap_to_release is not None:
            logging.info("Requesting static video release via executor...")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    executor, _release_static_video_blocking, cap_to_release
                )
            except Exception as e:
                logging.error(f"Error running static video release in executor: {e}")
        logging.info("Static video release process finished.")


def _process_frame_motion_detection_blocking(current_frame_color, previous_frame_gray):
    """
    Detects motion by comparing the current frame to the previous one.
    Returns the processed color frame and the new gray frame for the next iteration.
    """
    if current_frame_color is None:
        return None, None

    processed_frame = current_frame_color.copy()

    current_frame_gray = cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.GaussianBlur(current_frame_gray, MOTION_BLUR_KERNEL, 0)

    if previous_frame_gray is None:
        cv2.putText(
            processed_frame,
            "Initializing Motion...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return (
            processed_frame,
            current_frame_gray,
        )

    try:
        frame_delta = cv2.absdiff(previous_frame_gray, current_frame_gray)

        thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((5, 5), np.uint8)  # Kernel for dilation
        thresh = cv2.dilate(thresh, kernel, iterations=MOTION_DILATE_ITERATIONS)

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < MOTION_MIN_AREA:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                processed_frame,
                "Motion",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            motion_detected = True

        return processed_frame, current_frame_gray

    except Exception as e:
        logging.error(f"[Executor] Error in motion detection: {e}", exc_info=True)
        return current_frame_color, current_frame_gray


def _read_frame_blocking(cap_obj):
    if cap_obj is None or not cap_obj.isOpened():
        return None
    try:
        success, frame = cap_obj.read()
        return frame if success else None
    except Exception as e:
        logging.error(f"[Executor] Exception during frame read: {e}", exc_info=True)
        return None


async def check_and_manage_camera_task():
    global CAMERA_TASK
    subscribers_exist = bool(SUBSCRIPTIONS["normal_video"])
    if subscribers_exist:
        if CAMERA is None:
            if not await initialize_camera():
                return
        if CAMERA is not None and (CAMERA_TASK is None or CAMERA_TASK.done()):
            logging.info(
                "Live Camera ready. Starting live broadcast task (motion detection)..."
            )
            CAMERA_TASK = asyncio.create_task(broadcast_camera_frames())
        elif CAMERA is not None:
            logging.debug("Live broadcast task already running.")
    elif not subscribers_exist:
        if CAMERA is not None or (CAMERA_TASK and not CAMERA_TASK.done()):
            await release_camera()
        else:
            logging.debug("No live subscribers and camera/task already stopped.")


async def check_and_manage_static_video_task():
    global STATIC_VIDEO_TASK
    subscribers_exist = bool(SUBSCRIPTIONS["static_video"])
    if subscribers_exist:
        if not STATIC_VIDEO_PATH.is_file():
            return
        if STATIC_VIDEO_CAPTURE is None:
            if not await initialize_static_video():
                return
        if STATIC_VIDEO_CAPTURE is not None and (
            STATIC_VIDEO_TASK is None or STATIC_VIDEO_TASK.done()
        ):
            logging.info(
                "Static Video ready. Starting static broadcast task (motion detection)..."
            )
            STATIC_VIDEO_TASK = asyncio.create_task(broadcast_static_video_frames())
        elif STATIC_VIDEO_CAPTURE is not None:
            logging.debug("Static video broadcast task already running.")
    elif not subscribers_exist:
        if STATIC_VIDEO_CAPTURE is not None or (
            STATIC_VIDEO_TASK and not STATIC_VIDEO_TASK.done()
        ):
            await release_static_video()
        else:
            logging.debug("No static video subscribers and video/task already stopped.")


async def handle_websocket_message(ws, addr, message):
    try:
        data = json.loads(message)
        action = data.get("action")
        stream = data.get("stream")
        logging.debug(f"Received WS message from {addr}: {data}")
        stream_updated = False
        target_stream_set = SUBSCRIPTIONS.get(stream)

        if target_stream_set is not None:
            if action == "subscribe":
                if ws not in target_stream_set:
                    other_stream = (
                        "static_video" if stream == "normal_video" else "normal_video"
                    )
                    if ws in SUBSCRIPTIONS[other_stream]:
                        SUBSCRIPTIONS[other_stream].discard(ws)
                        logging.info(
                            f"Client {addr} auto-unsubscribed from '{other_stream}'."
                        )
                        if other_stream == "normal_video":
                            await check_and_manage_camera_task()
                        else:
                            await check_and_manage_static_video_task()
                    target_stream_set.add(ws)
                    logging.info(
                        f"Client {addr} subscribed to '{stream}'. Total: {len(target_stream_set)}"
                    )
                    stream_updated = True
            elif action == "unsubscribe":
                if ws in target_stream_set:
                    target_stream_set.discard(ws)
                    logging.info(
                        f"Client {addr} unsubscribed from '{stream}'. Total: {len(target_stream_set)}"
                    )
                    stream_updated = True
            else:
                logging.warning(
                    f"Unknown action '{action}' for stream '{stream}' from {addr}"
                )
        else:
            logging.warning(f"Unknown stream '{stream}' from {addr}: {data}")

        if stream_updated:
            if stream == "normal_video":
                await check_and_manage_camera_task()
            elif stream == "static_video":
                await check_and_manage_static_video_task()
    except json.JSONDecodeError:
        logging.error(f"Non-JSON message from {addr}: {message}")
    except Exception as e:
        logging.error(
            f"Error processing message '{message}' from {addr}: {e}", exc_info=True
        )


async def websocket_handler(request):
    addr_str = str(request.remote) if request.remote else "unknown"
    ws = web.WebSocketResponse()
    if not ws.can_prepare(request).ok:
        logging.error(f"WS prep failed for {addr_str}")
        return web.Response(status=400, text="WebSocket upgrade failed")
    await ws.prepare(request)
    logging.info(f"WebSocket connection established from {addr_str}")
    CLIENTS.add(ws)
    ws.addr_str = addr_str
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await handle_websocket_message(ws, addr_str, msg.data)
            elif msg.type == web.WSMsgType.BINARY:
                logging.warning(f"Unexpected binary message from {addr_str}.")
            elif msg.type == web.WSMsgType.ERROR:
                logging.error(
                    f"WS connection for {addr_str} closed with exception {ws.exception()}"
                )
    except asyncio.CancelledError:
        logging.info(f"WS task for {addr_str} cancelled.")
        raise
    except Exception as e:
        logging.error(f"Error in WS handler for {addr_str}: {e}", exc_info=True)
    finally:
        logging.info(f"WebSocket connection closed for {addr_str}")
        CLIENTS.discard(ws)
        client_was_live_subscriber = False
        if ws in SUBSCRIPTIONS["normal_video"]:
            SUBSCRIPTIONS["normal_video"].discard(ws)
            client_was_live_subscriber = True
        client_was_static_subscriber = False
        if ws in SUBSCRIPTIONS["static_video"]:
            SUBSCRIPTIONS["static_video"].discard(ws)
            client_was_static_subscriber = True
        if client_was_live_subscriber:
            asyncio.create_task(check_and_manage_camera_task())
        if client_was_static_subscriber:
            asyncio.create_task(check_and_manage_static_video_task())
    return ws


async def broadcast_camera_frames():
    """Continuously reads (live cam), performs motion detection, encodes, and sends."""
    logging.info("LIVE camera broadcast task started (motion detection).")
    loop = asyncio.get_running_loop()
    frame_count = 0
    last_log_time = time.monotonic()
    previous_frame_gray = None

    while True:
        start_time = time.monotonic()
        cam_current = CAMERA
        if cam_current is None:
            await asyncio.sleep(0.5)
            continue

        try:
            current_frame_color = await loop.run_in_executor(
                executor, _read_frame_blocking, cam_current
            )
            if CAMERA is None:
                logging.debug("Live camera released during read.")
                break
            if current_frame_color is None:
                logging.warning("Failed to capture frame from live camera.")
                await asyncio.sleep(0.1)
                continue

            if len(current_frame_color.shape) == 2:
                current_frame_color = cv2.cvtColor(
                    current_frame_color, cv2.COLOR_GRAY2BGR
                )
            elif current_frame_color.shape[2] == 4:
                current_frame_color = cv2.cvtColor(
                    current_frame_color, cv2.COLOR_BGRA2BGR
                )

            processed_frame_color, next_previous_frame_gray = (
                await loop.run_in_executor(
                    executor,
                    _process_frame_motion_detection_blocking,
                    current_frame_color,
                    previous_frame_gray,
                )
            )
            if CAMERA is None:
                logging.debug("Live camera released during processing.")
                break

            previous_frame_gray = (
                next_previous_frame_gray.copy()
                if next_previous_frame_gray is not None
                else None
            )

            if processed_frame_color is None:
                logging.warning("Motion detection failed for live frame.")
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, buffer = cv2.imencode(".jpg", processed_frame_color, encode_param)
            if not ret:
                logging.warning("Failed to encode live frame to JPEG.")
                continue
            frame_bytes = buffer.tobytes()

            subscribers = list(SUBSCRIPTIONS["normal_video"])
            if subscribers:
                tasks = [
                    asyncio.create_task(safe_send_binary(ws, frame_bytes))
                    for ws in subscribers
                    if not ws.closed
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            frame_count += 1
            cycle_time = time.monotonic() - start_time
            if time.monotonic() - last_log_time >= 5.0:
                fps = frame_count / (time.monotonic() - last_log_time)
                logging.info(
                    f"Live broadcast FPS: {fps:.1f} (motion detection) (last cycle: {cycle_time*1000:.1f} ms)"
                )
                frame_count = 0
                last_log_time = time.monotonic()

            processing_time = time.monotonic() - start_time
            sleep_duration = max(0, FRAME_DELAY - processing_time)
            await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            logging.info("Live camera broadcast task cancelled.")
            break
        except Exception as e:
            logging.error(f"Error in live camera broadcast loop: {e}", exc_info=True)
            await asyncio.sleep(1)

    logging.info("Live camera broadcast task finished.")


async def broadcast_static_video_frames():
    """Continuously reads (static video), performs motion detection, encodes, and sends."""
    logging.info("STATIC video broadcast task started (motion detection).")
    loop = asyncio.get_running_loop()
    global STATIC_VIDEO_CAPTURE
    frame_count = 0
    last_log_time = time.monotonic()
    previous_frame_gray = None
    retries = 0
    max_retries = 3

    while True:
        start_time = time.monotonic()
        cap_current = STATIC_VIDEO_CAPTURE
        if cap_current is None:
            await asyncio.sleep(0.5)
            continue

        try:
            current_frame_color = await loop.run_in_executor(
                executor, _read_frame_blocking, cap_current
            )

            if current_frame_color is None:
                logging.info(
                    "End of static video file reached or read error. Attempting to reopen/loop..."
                )
                await release_static_video()
                initialized = await initialize_static_video()
                if initialized:
                    logging.info("Static video re-initialized for looping.")
                    previous_frame_gray = None
                    retries = 0
                    continue
                else:
                    logging.error("Failed to re-initialize static video for looping.")
                    retries += 1
                    if retries > max_retries:
                        break
                    await asyncio.sleep(2)
                    continue

            if STATIC_VIDEO_CAPTURE is None:
                logging.debug("Static video released during read.")
                break

            if len(current_frame_color.shape) == 2:
                current_frame_color = cv2.cvtColor(
                    current_frame_color, cv2.COLOR_GRAY2BGR
                )
            elif current_frame_color.shape[2] == 4:
                current_frame_color = cv2.cvtColor(
                    current_frame_color, cv2.COLOR_BGRA2BGR
                )

            processed_frame_color, next_previous_frame_gray = (
                await loop.run_in_executor(
                    executor,
                    _process_frame_motion_detection_blocking,
                    current_frame_color,
                    previous_frame_gray,
                )
            )
            if STATIC_VIDEO_CAPTURE is None:
                logging.debug("Static video released during processing.")
                break

            previous_frame_gray = (
                next_previous_frame_gray.copy()
                if next_previous_frame_gray is not None
                else None
            )

            if processed_frame_color is None:
                logging.warning("Motion detection failed for static frame.")
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, buffer = cv2.imencode(".jpg", processed_frame_color, encode_param)
            if not ret:
                logging.warning("Failed to encode static frame to JPEG.")
                continue
            frame_bytes = buffer.tobytes()

            subscribers = list(SUBSCRIPTIONS["static_video"])
            if subscribers:
                tasks = [
                    asyncio.create_task(safe_send_binary(ws, frame_bytes))
                    for ws in subscribers
                    if not ws.closed
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            frame_count += 1
            cycle_time = time.monotonic() - start_time
            if time.monotonic() - last_log_time >= 5.0:
                fps = frame_count / (time.monotonic() - last_log_time)
                logging.info(
                    f"Static broadcast FPS: {fps:.1f} (motion detection) (last cycle: {cycle_time*1000:.1f} ms)"
                )
                frame_count = 0
                last_log_time = time.monotonic()

            processing_time = time.monotonic() - start_time
            sleep_duration = max(0, FRAME_DELAY - processing_time)
            await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            logging.info("Static video broadcast task cancelled.")
            break
        except Exception as e:
            logging.error(f"Error in static video broadcast loop: {e}", exc_info=True)
            await asyncio.sleep(1)

    logging.info("Static video broadcast task finished.")


async def safe_send_binary(ws, data):
    addr = ws.addr_str if hasattr(ws, "addr_str") else "unknown_addr_send"
    try:
        await ws.send_bytes(data)
    except (ConnectionResetError, asyncio.CancelledError, RuntimeError) as e:
        logging.warning(f"Client {addr} disconnected during send: {type(e).__name__}")
        CLIENTS.discard(ws)
        was_live = False
        was_static = False
        if ws in SUBSCRIPTIONS["normal_video"]:
            SUBSCRIPTIONS["normal_video"].discard(ws)
            was_live = True
        if ws in SUBSCRIPTIONS["static_video"]:
            SUBSCRIPTIONS["static_video"].discard(ws)
            was_static = True
        if was_live:
            asyncio.create_task(check_and_manage_camera_task())
        if was_static:
            asyncio.create_task(check_and_manage_static_video_task())
        if not isinstance(e, asyncio.CancelledError):
            raise
    except Exception as e:
        logging.error(f"Unexpected error sending frame to {addr}: {e}")
        CLIENTS.discard(ws)
        was_live = False
        was_static = False
        if ws in SUBSCRIPTIONS["normal_video"]:
            SUBSCRIPTIONS["normal_video"].discard(ws)
            was_live = True
        if ws in SUBSCRIPTIONS["static_video"]:
            SUBSCRIPTIONS["static_video"].discard(ws)
            was_static = True
        if was_live:
            asyncio.create_task(check_and_manage_camera_task())
        if was_static:
            asyncio.create_task(check_and_manage_static_video_task())
        raise


async def handle_index(request):
    index_path = STATIC_FILES_DIR / "plan.html"
    return (
        web.FileResponse(index_path)
        if index_path.is_file()
        else web.Response(status=404, text="plan.html not found")
    )


async def handle_static(request):
    req_path_str = request.match_info.get("filename", "")
    if not req_path_str:
        return web.Response(status=404, text="File not specified")
    try:
        req_path = Path(req_path_str)
        if ".." in req_path.parts:
            return web.Response(status=403, text="Forbidden")
        file_path = (STATIC_FILES_DIR / req_path).resolve()
        allowed_base = STATIC_FILES_DIR.resolve()
        is_safe = False
        if file_path.is_relative_to(allowed_base):
            if file_path.parent == allowed_base or file_path.parent == (
                allowed_base / "static"
            ):
                is_safe = True
        if not is_safe:
            return web.Response(status=403, text="Forbidden")
        if file_path.is_file():
            content_type = None
            if file_path.suffix == ".css":
                content_type = "text/css"
            elif file_path.suffix == ".js":
                content_type = "application/javascript"
            elif file_path.suffix == ".mp4":
                content_type = "video/mp4"
            elif file_path.suffix == ".svg":
                content_type = "image/svg+xml"
            return web.FileResponse(
                file_path,
                chunk_size=256 * 1024,
                headers={"Content-Type": content_type} if content_type else None,
            )
        else:
            return web.Response(status=404, text=f"{req_path_str} not found")
    except ValueError:
        return web.Response(status=400, text="Bad Request")
    except Exception as e:
        logging.error(f"Static file error {req_path_str}: {e}", exc_info=True)
        return web.Response(status=500)


async def setup_app():
    logging.info("OpenCV Motion Detection will be used (No ML Model).")  # Updated log
    if not STATIC_VIDEO_PATH.is_file():
        logging.warning(
            f"Static video file not found: {STATIC_VIDEO_PATH}. Static streaming disabled."
        )
    else:
        logging.info(f"Static video file found: {STATIC_VIDEO_PATH}")

    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/", handle_index)
    app.router.add_get("/{filename:.+}", handle_static)
    app.on_cleanup.append(cleanup_app)
    return app


async def cleanup_app(app_instance):
    logging.info("Server shutting down, cleaning up resources...")
    await release_camera()
    await release_static_video()
    logging.info("Cleanup: Shutting down thread pool executor...")
    executor.shutdown(wait=True, cancel_futures=True)
    logging.info("Cleanup: Executor shut down.")


async def run_server():
    app = await setup_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    logging.info(f"-----------------------------------------------------")
    logging.info(f"Starting AIOHTTP Server with OpenCV Motion Detection")
    logging.info(f"Live Webcam ({WEBCAM_INDEX}) will be initialized ON DEMAND.")
    if STATIC_VIDEO_PATH.is_file():
        logging.info(
            f"Static Video ({STATIC_VIDEO_PATH.name}) streaming available ON DEMAND."
        )
    else:
        logging.warning(
            f"Static Video ({STATIC_VIDEO_PATH.name}) NOT FOUND - streaming disabled."
        )
    logging.info(f"Serving on http://{HOST}:{PORT} (and ws://{HOST}:{PORT}/ws)")
    logging.info(f"Access via local IP: http://<YOUR_LOCAL_IP>:{PORT}")
    logging.info(f"Press Ctrl+C to stop.")
    logging.info(f"-----------------------------------------------------")
    await site.start()
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Shutdown signal received.")
    finally:
        logging.info("Stopping site and runner...")
        await site.stop()
        await runner.cleanup()
        logging.info("Site and runner stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt caught in main, allowing cleanup.")
    finally:
        logging.info("Server process ending.")
