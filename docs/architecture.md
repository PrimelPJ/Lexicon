# Lexicon architecture notes

## From words to a map pose

```
instruction ──parse──▶ phrases ──OWL-ViT──▶ box (u,v) in image
                                             │
                          depth patch median │  robust z (metres)
                                             ▼
                    intrinsics back-projection: point in camera optical frame
                                             │
                                   tf2 camera -> map (4x4)
                                             ▼
                          point in map frame ──approach──▶ PoseStamped goal
```

## Grounding math

Back-projection uses the pinhole model with the ROS optical frame convention
(z forward, x right, y down):

```
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth
```

`cx, cy, fx, fy` come from `CameraInfo.k`. The point is then multiplied by the
4x4 camera-to-map transform from tf2. The approach pose stops `standoff` metres
short along the robot-to-target ray and faces the target. All of this lives in
`grounding.py` and is unit tested without ROS.

## Concurrency model

The action server is simultaneously:
- an action server (FindObject),
- a service client (Ground),
- an action client (Nav2 NavigateToPose).

If these ran on the default single-threaded executor, awaiting the Ground future
inside the FindObject execute callback would block the one thread that also has
to process the service response, and the node would deadlock. The fix, used
here, is a ReentrantCallbackGroup plus a MultiThreadedExecutor, so the await
yields and the response callback runs on another thread. This is the standard
ROS2 pattern for composing async calls and is worth being able to explain.

## Lifecycle rationale

The VLM is hundreds of megabytes to gigabytes. Loading it in `__init__` would
make every launch slow and every idle robot memory-hungry. As a lifecycle node:
- `on_configure` sets up subscriptions, tf, the service, and the publisher, but
  does not load weights.
- `on_activate` loads the model. `on_deactivate` frees it.
- The bringup launch drives configure then activate automatically via lifecycle
  event handlers.

## Failure modes and how they surface

- No camera frame yet: Ground returns success=false with a clear message.
- Phrase not found: Ground returns the phrases it tried.
- Matched in image but depth invalid on the target: reported distinctly, since
  it points at a sensor/geometry issue rather than a perception miss.
- tf lookup fails: reported, usually a frames or clock (use_sim_time) problem.
- Nav2 unavailable or rejects the goal: the action aborts with the reason.

Returning specific, actionable failures is part of making the stack debuggable.
