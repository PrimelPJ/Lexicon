# Lexicon

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/PrimelPJ/lexicon/actions/workflows/test.yml/badge.svg)

Tell a ROS2 robot "go to the red chair" and it finds the chair using a vision-language model, figures out where that is in 3D space, and drives there with Nav2. No fixed list of objects, no retraining required: the detector accepts plain text, so the vocabulary is open.

> **Why this project.** The interesting part is not calling a model. It is wiring a vision-language model into a real ROS2 stack correctly: a lifecycle-managed perception node, a custom action interface, an action server that composes a service call and a Nav2 action, tf2 for turning a pixel into a map pose, and inference that never blocks the executor. This is built the way a ROS2 team would build it.

The vision model half runs with zero ROS setup via `scripts/detect_image.py`, and the core logic is unit tested.

---

## How it works

Lexicon gives the robot a pair of eyes that understand words. You say the name of a thing, the robot spots it in the camera view, figures out where that spot is in the real room, and drives over to it.

The key ideas, in plain terms:

- **Open vocabulary.** The robot is not limited to a fixed list of objects it was trained on. The vision model accepts plain text, so "the red chair", "the fire extinguisher", or "the blue toolbox" all work without retraining. You can send the robot to things nobody thought to label in advance.

- **Grounding (from a pixel to a map coordinate).** Spotting the chair in the image is only half the job. To drive there the robot needs to know where that pixel is in the real room. It uses the depth camera to measure how far away the chair is, combined with its knowledge of where the camera is mounted and where it is standing, to convert that pixel into a real coordinate on its map.

- **Approach, do not collide.** The robot targets a spot a set distance short of the object and faces it, so it pulls up in front of the chair rather than driving into it.

- **Load the big model only when needed.** The vision-language model is large. The robot loads it into memory only when activated for a task and frees it when idle, so an idle robot is not wasting memory.

- **Navigation is a task, not a one-shot call.** Driving somewhere takes time, produces live progress updates ("2 metres to go"), and must be cancellable. The command is run as a ROS2 action, which supports all of that, rather than a simple blocking call.

---

## Step-by-step: what happens when you say "go to the red chair"

1. The action server receives the sentence and extracts the search phrases: "red chair" and, as a fallback, "chair".
2. It asks the perception node to find those phrases. The vision-language model scans the current camera frame and returns a bounding box around the best match.
3. The perception node reads the depth inside that box, converts it to a 3D point, and uses tf2 to express that point in the robot's map frame. It sends back a map coordinate for the chair.
4. The action server computes a goal a little short of the chair, facing it, and sends that goal to Nav2.
5. Nav2 plans a path and drives there. Progress is relayed back as feedback the whole way ("navigating, 1.4 metres remaining").
6. On arrival the action reports success, or a clear reason for failure (chair not visible, no valid depth, navigation blocked).

Meanwhile a green marker for the chair appears in RViz, so an operator can see exactly what the robot understood the command to mean.

### The moving parts

| Module | What it does |
|---|---|
| `command_parser` | Turns a sentence into open-vocabulary search phrases |
| `open_vocab_detector` | The vision-language model that finds text queries in an image |
| `grounding` | Converts a pixel plus depth into a map coordinate |
| `detector_node` | Lifecycle perception node: detects, grounds, and serves results |
| `find_object_server` | Action server: parse, ground, then drive via Nav2 |
| `markers` | Draws the understood target in RViz |

---

## Architecture

```
  "go to the red chair"
          |
          v
  +-------------------------+         FindObject.action (goal/feedback/result)
  |  find_object_server     |<----------------- ros2 action send_goal
  |  (action server)        |
  |   1. parse instruction -+---> command_parser  -> ["red chair", "chair"]
  |   2. call Ground service|
  |   3. approach pose (tf2)|
  |   4. NavigateToPose ----+---> Nav2 (navigate_to_pose action)
  |   5. relay feedback     |
  +-----------+-------------+
              |  Ground.srv (phrases -> 3D poses in map)
              v
  +-------------------------+   synced RGB + depth + CameraInfo
  |  lexicon_detector       |<-----------------------------------
  |  (LIFECYCLE node)       |
  |   OWL-ViT open-vocab ---+---> bounding boxes for the text query
  |   depth + intrinsics ---+---> 3D point in camera frame
  |   tf2 camera -> map ----+---> PoseStamped in map frame
  |   RViz MarkerArray      +--> /lexicon/markers
  +-------------------------+

  lifecycle:  unconfigured --configure--> inactive --activate--> active
              (VLM weights load on activate, freed on deactivate)
```

See `docs/architecture.md` for the grounding math and the concurrency model.

---

## ROS2 concepts demonstrated

This project is a hands-on tour of the parts of ROS2 that production teams rely on:

- **Lifecycle (managed) nodes.** The detector loads a large model only on `activate` and frees it on `deactivate`, with a launch file that drives the transitions automatically.
- **Custom interfaces.** A separate `lexicon_interfaces` ament_cmake package defines a `FindObject.action` and a `Ground.srv`, generated with `rosidl_generate_interfaces`.
- **Actions for long-running tasks.** Navigation is an action (with feedback and cancellation), not a service. The server also acts as a Nav2 action client.
- **Service composition.** The action server calls the detector's Ground service and awaits the result inside an async execute callback.
- **Concurrency done right.** Reentrant callback groups plus a MultiThreadedExecutor so awaiting a service or action future does not deadlock.
- **tf2.** Camera-optical-frame points are transformed into the map frame using live transforms; the approach pose is computed from the robot's current pose.
- **Sensor sync and QoS.** RGB, depth, and CameraInfo are combined with an ApproximateTimeSynchronizer under sensor-data QoS.
- **Visualization.** Grounded targets are published as an RViz MarkerArray.
- **Testable core.** Grounding geometry and instruction parsing are pure modules with a passing pytest suite.

---

## Repository layout

```
lexicon/
  ros2_ws/src/
    lexicon_interfaces/          ament_cmake: FindObject.action, Ground.srv
    lexicon/                     ament_python
      lexicon/
        open_vocab_detector.py   OWL-ViT wrapper (swap in Grounding DINO / YOLO-World)
        detector_node.py         LIFECYCLE node: sync RGB-D, detect, ground, serve
        find_object_server.py    action server: parse -> ground -> Nav2
        grounding.py             pure geometry (pixel+depth+tf -> map pose)  [tested]
        command_parser.py        instruction -> open-vocab phrases           [tested]
        markers.py               RViz markers
      launch/bringup.launch.py   detector (auto-activated) + action server
      config/params.yaml
      test/                      pytest suite for the pure logic
  scripts/
    detect_image.py             VLM demo on one image, no ROS required
    demo_query.py               send a FindObject goal from the CLI
  docs/architecture.md
```

---

## Quickstart

### Part 1: try the vision model, no ROS needed

```bash
pip install -r requirements.txt
cd scripts
python detect_image.py --image your_room.jpg --instruction "go to the red chair"
# writes detections.jpg with bounding boxes, prints scores and parsed queries
```

### Part 2: run the pure logic tests

```bash
cd ros2_ws/src/lexicon
pip install pytest numpy
python -m pytest test/ -q      # grounding geometry + instruction parsing
```

### Part 3: the full stack in simulation

Prerequisites: ROS2 Humble or Jazzy, Nav2, and a TurtleBot3 (or any robot with an RGB-D camera and a Nav2 map).

```bash
# build (interfaces first, then the Python package)
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

pip install -r ../requirements.txt   # into the same environment

# terminal 1: robot + Nav2 (TurtleBot3 example)
export TURTLEBOT3_MODEL=waffle       # waffle has a depth camera
ros2 launch nav2_bringup tb3_simulation_launch.py

# terminal 2: Lexicon (detector auto-activates, action server starts)
ros2 launch lexicon bringup.launch.py

# terminal 3: give it a command
ros2 run lexicon demo_query.py "go to the nearest chair"
# or the raw CLI:
# ros2 action send_goal /find_object lexicon_interfaces/action/FindObject "{instruction: 'find the backpack'}"
```

Watch the grounded target appear in RViz on `/lexicon/markers` and the robot drive to it.

---

## Design decisions

**Open vocabulary is the point.** A fixed 80-class detector cannot handle "the red toolbox" or "the fire extinguisher" without retraining. OWL-ViT accepts the phrase directly, so the set of things the robot can navigate to is open. That is what makes the language interface actually useful.

**Use the median depth over the detection box, not a single pixel.** A single depth sample often lands on holes or edges. Grounding takes the median depth over the box interior, which is robust to those. See `grounding.median_depth`.

**Approach, do not collide.** The action server computes a goal that stops a configurable distance short of the target and faces it, rather than driving to the object's exact coordinates.

**The VLM never blocks the control loop.** Detection runs inside a service call handled on a reentrant callback group under a multithreaded executor. The action server awaits it asynchronously. Nothing in the perception path stalls the node's other callbacks.

**Load the model at a known time.** As a lifecycle node, the detector loads weights on activate and frees them on deactivate, so an idle robot is not holding gigabytes and startup cost is explicit, not hidden inside the first frame.

**Swap the model freely.** `OpenVocabDetector.detect()` is the only interface the rest of the stack depends on. Grounding DINO and YOLO-World implement the same text-to-boxes idea and drop in without touching the node, the service, or the action server.

---

## Extending it

- Add a `Search.action` that spins or patrols until the target is seen, then hands off to `find_object`.
- Add a small LLM in `command_parser.llm_extract` for multi-step instructions ("go to the kitchen and find a mug").
- Compose the detector as a component in a shared container with the camera driver to cut serialization overhead.
- Publish `vision_msgs/Detection3DArray` for downstream consumers.

---

## Limitations

- **Occluded and off-frame targets.** The detector queries a single camera frame. If the object is behind another object or outside the field of view, the action aborts. There is no search or patrol behaviour to recover.
- **No multi-instance disambiguation.** When multiple instances match the query (e.g. two chairs in view), only the highest-confidence candidate is returned (or the nearest one if "nearest" is in the instruction). Others are silently ignored.
- **Static goal, no live re-grounding.** The 3D target position is computed once when the command arrives. If the target moves after the goal is sent to Nav2, the robot drives to the original location.
- **Dynamic obstacles mid-approach.** Nav2 replans around obstacles that appear en route, but if the path becomes permanently blocked the action fails with no retry or alternative approach strategy.
- **Vision model quality.** Detection confidence drives every downstream step. Low-light scenes, extreme viewpoints, or ambiguous queries can produce no detections or an incorrectly grounded pose. There is no fallback query or re-prompt logic.
- **Depth sensor limitations.** Grounding relies on valid depth pixels inside the detection box. Transparent, highly reflective, or out-of-range objects often produce no valid depth, causing `median_depth` to return `None` and the action to abort.

---

## License

MIT. See `LICENSE`.
