# Lexicon

Tell a ROS2 robot "go to the red chair" and it finds the chair with an
open-vocabulary vision-language model, works out where that is in 3D, and
navigates there with Nav2. No fixed class list, no retraining to add an object:
the vocabulary is open because the detector takes free text.

> **Why this project.** The interesting part is not calling a model. It is
> wiring a VLM into a real ROS2 stack correctly: a lifecycle-managed perception
> node, a custom action interface, an action server that composes a service call
> and a Nav2 action, tf2 for turning a pixel into a map pose, and inference that
> never blocks the executor. This is built the way a ROS2 team would want it
> built.

Runs the VLM half with zero ROS setup through `scripts/detect_image.py`, and the
pure logic is unit tested.

---

## How it works in plain terms

Lexicon gives the robot a pair of eyes that understand words. You say the name of
a thing, the robot spots it in the camera view, figures out where that spot is in
the real room, and drives over to it.

The ideas that make this work, in everyday language:

- **Open vocabulary.** The robot is not limited to a fixed list of objects it was
  trained on. The vision model takes plain text, so "the red chair", "the fire
  extinguisher", or "the blue toolbox" all work without retraining anything. You
  can send the robot after things nobody thought of in advance.

- **Grounding (from a spot in the picture to a spot on the map).** Finding the
  chair in the image is only half the job. To drive there the robot has to know
  where that pixel is in the room. It uses the depth camera to learn how far away
  the chair is, and its own knowledge of where the camera is mounted and where it
  is standing, to convert that pixel into a real coordinate on its map.

- **Approach, do not crash into it.** The robot aims for a spot a set distance
  short of the target and turns to face it, so it pulls up in front of the chair
  rather than driving into it.

- **Load the big model only when needed.** The vision-language model is large.
  The robot loads it into memory only when the system is switched on for a task
  and frees it when idle, so an idle robot is not wasting memory.

- **Navigation is a task, not a question.** Driving somewhere takes time, needs
  live progress ("2 metres to go"), and must be cancelable. So the command is run
  as a ROS2 action (which supports all of that) rather than a simple call that
  just blocks until it finishes.

---

## Follow one command

The life of "go to the red chair":

1. The action server receives the sentence and strips it down to what matters:
   the phrases "red chair" and, as a fallback, "chair".
2. It asks the perception node to find those phrases. The vision-language model
   scans the current camera frame and returns a box around the best match.
3. The perception node reads the depth at that box, converts the box into a 3D
   point, and uses tf2 to express that point on the robot's map. It sends back a
   map coordinate for the chair.
4. The action server computes a goal a little short of the chair, facing it, and
   hands that goal to Nav2.
5. Nav2 plans a path and drives there. Its progress is relayed back as feedback
   the whole way ("navigating, 1.4 metres remaining").
6. On arrival the action reports success, or a clear reason if anything failed
   (chair not visible, no valid depth, navigation blocked).

Meanwhile a green marker for the chair appears in RViz, so an operator can see
exactly what the robot understood the command to mean.

### The moving parts, one line each

- **command_parser**: turns a sentence into open-vocabulary phrases.
- **open_vocab_detector**: the vision-language model that finds text queries in an image.
- **grounding**: converts a pixel plus depth into a map coordinate (tf2 math).
- **detector_node**: the lifecycle perception node that detects and grounds, and serves it.
- **find_object_server**: the action server that parses, grounds, and drives via Nav2.
- **markers**: draws the understood target in RViz.

---

## Architecture

```
  "go to the red chair"
          │
          ▼
  ┌─────────────────────────┐         FindObject.action (goal/feedback/result)
  │  find_object_server      │◀───────────────── ros2 action send_goal
  │  (action server)         │
  │   1. parse instruction ──┼──▶ command_parser  -> ["red chair", "chair"]
  │   2. call Ground service │
  │   3. approach pose (tf2) │
  │   4. NavigateToPose ─────┼──▶ Nav2 (navigate_to_pose action)
  │   5. relay feedback      │
  └───────────┬─────────────┘
              │ Ground.srv (phrases -> 3D poses in map)
              ▼
  ┌─────────────────────────┐   synced RGB + depth + CameraInfo
  │  lexicon_detector        │◀──────────────────────────────────
  │  (LIFECYCLE node)        │
  │   OWL-ViT open-vocab ────┼──▶ boxes for the text query   [VLM]
  │   depth + intrinsics ────┼──▶ 3D point in camera frame
  │   tf2 camera -> map ─────┼──▶ PoseStamped in map frame
  │   RViz MarkerArray       │──▶ /lexicon/markers
  └─────────────────────────┘

  lifecycle:  unconfigured --configure--> inactive --activate--> active
              (VLM weights load on activate, free on deactivate)
```

See `docs/architecture.md` for the grounding math and the concurrency model.

---

## ROS2 concepts demonstrated

This project is deliberately a tour of the parts of ROS2 a team actually relies
on:

- **Lifecycle (managed) nodes.** The detector loads a large model only on
  `activate` and frees it on `deactivate`, with a launch file that drives the
  transitions automatically.
- **Custom interfaces.** A separate `lexicon_interfaces` ament_cmake package
  defines a `FindObject.action` and a `Ground.srv`, generated with
  `rosidl_generate_interfaces`.
- **Actions for long tasks.** Navigation is an action (feedback, cancelation),
  not a service. The server also acts as a Nav2 action client.
- **Service composition.** The action server calls the detector's Ground
  service and awaits the result inside an async execute callback.
- **Concurrency done right.** Reentrant callback groups plus a
  MultiThreadedExecutor so awaiting a service or action future does not deadlock.
- **tf2.** Camera-optical-frame points are transformed into the map frame using
  live transforms; the approach pose is computed from the robot's current pose.
- **Sensor sync + QoS.** RGB, depth, and CameraInfo are combined with an
  ApproximateTimeSynchronizer under sensor-data QoS.
- **Visualization.** Grounded targets are published as an RViz MarkerArray.
- **Testable core.** Grounding geometry and instruction parsing are pure modules
  with a passing pytest suite.

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
    detect_image.py             VLM demo on one image, no ROS
    demo_query.py               send a FindObject goal from the CLI
  docs/architecture.md
```

---

## Quickstart

### Part 1: the VLM, no ROS needed

```bash
pip install torch transformers pillow numpy
cd scripts
python detect_image.py --image your_room.jpg --instruction "go to the red chair"
# writes detections.jpg with boxes, prints scores and parsed queries
```

### Part 2: the pure logic tests

```bash
cd ros2_ws/src/lexicon
pip install pytest numpy
python -m pytest test/ -q      # grounding geometry + instruction parsing
```

### Part 3: the full stack in simulation

Prerequisites: ROS2 Humble or Jazzy, Nav2, and a TurtleBot3 (or any robot with
an RGB-D camera and a Nav2 map).

```bash
# build (interfaces first, then the python package)
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

pip install torch transformers pillow numpy   # into the same environment

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

Watch the grounded target appear in RViz on `/lexicon/markers` and the robot
drive to it.

---

## Design decisions

**Open vocabulary is the point.** A fixed 80-class detector cannot handle "the
red toolbox" or "the fire extinguisher" without retraining. OWL-ViT takes the
phrase directly, so the set of things the robot can be sent to is open. That is
what makes the language interface actually useful.

**Ground with a depth patch median, not one pixel.** A single depth sample lands
on holes and edges. The grounding takes the median depth over the box interior,
which is robust to those. See `grounding.median_depth`.

**Approach, do not collide.** The action server computes a goal that stops a
configurable standoff short of the target and faces it, rather than driving to
the object's exact coordinates.

**The VLM never blocks the control loop.** Detection runs inside a service call
handled on a reentrant callback group under a multithreaded executor. The action
server awaits it asynchronously. Nothing in the perception path stalls the node's
other callbacks.

**Load the model at a known time.** As a lifecycle node, the detector loads
weights on activate and frees them on deactivate, so an idle robot is not holding
gigabytes and startup cost is explicit, not hidden on the first frame.

**Swap the model freely.** `OpenVocabDetector.detect()` is the only VLM contract.
Grounding DINO and YOLO-World implement the same text-to-boxes idea and drop in
without touching the node, the service, or the action server.

## Extending it

- Add a `Search.action` that spins or patrols until the target is seen, then
  hands off to `find_object`.
- Add a small LLM in `command_parser.llm_extract` for multi-step instructions
  ("go to the kitchen and find a mug").
- Compose the detector as a component in a shared container with the camera
  driver to cut serialization overhead.
- Publish `vision_msgs/Detection3DArray` for downstream consumers.

## License

MIT. See `LICENSE`.
