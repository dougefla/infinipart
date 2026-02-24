# InfiniPart

Multi-view articulated object rendering pipeline built on [Infinite-Mobility](https://github.com/OpenRobotLab/Infinite-Mobility). Generates procedural articulated 3D objects and renders multi-view motion videos with configurable animation modes.

## Features

- **16 object categories**: BeverageFridge, Microwave, Oven, Toilet, KitchenCabinet, Window, LiteDoor, OfficeChair, Tap, Lamp, Pot, Bottle, Dishwasher, BarChair, Pan, TV
- **Per-category animation modes (animodes)**: Independently control joint subsets (e.g., door only, drawers only, all joints)
- **32 camera views per animation**: 16 fixed hemisphere views + 8 back-to-front orbits + 8 front-hemisphere sweeps
- **Moving camera support**: Animated cameras that orbit or sweep during rendering
- **Batch pipeline**: Multi-seed × multi-animode × multi-view × multi-GPU parallel rendering

## Prerequisites

- **Blender 3.6** (headless, Linux x64)
- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (tested on L20X 143GB)
- **ffmpeg** for video encoding

## Setup

```bash
# 1. Download Blender 3.6
wget https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
tar xf blender-3.6.0-linux-x64.tar.xz

# 2. Clone this repo
git clone https://github.com/AuroraRyan0301/infinipart.git
cd infinipart

# 3. Install infinigen dependencies (uses Blender's bundled Python)
# See docs/Installation.md for details

# 4. Update paths in scripts:
#    - BLENDER path in batch_generate_all.py and render_articulation.py
#    - BASE_DIR to your repo root
#    - Envmap HDR path (--envmap flag)
```

## Quick Start

### Generate assets for one factory
```bash
CUDA_VISIBLE_DEVICES=0 blender --background --python-use-system-env \
  --python infinigen_examples/generate_individual_assets.py -- \
  --output_folder outputs/OvenFactory -f OvenFactory -n 1 --seed 0
```

### Render single factory/seed/animode
```bash
CUDA_VISIBLE_DEVICES=0 blender --background --python-use-system-env \
  --python render_articulation.py -- \
  --factory OvenFactory --seed 0 --device 0 \
  --output_dir outputs/motion_videos/OvenFactory/0 \
  --resolution 512 --samples 32 --duration 4.0 --fps 30 \
  --animode 0 --skip_bg \
  --views hemi_00 hemi_01 hemi_02 hemi_03 \
  --moving_views orbit_00 sweep_00
```

### Batch pipeline (generate + render all)
```bash
# Generate 10 seeds + render all animodes × 32 views on 4 GPUs
python batch_generate_all.py --n_seeds 10 --n_gpus 4 --no_split

# Render only (assets already generated)
python batch_generate_all.py --n_seeds 10 --n_gpus 4 --render_only --no_split

# Single factory
python batch_generate_all.py --n_seeds 10 --n_gpus 4 --factory OvenFactory --render_only --no_split
```

## Architecture

### Animation Modes (animodes)

Each factory defines animation modes that select specific joint subsets:

| Factory | Animode 0 | Animode 1 | Animode 2 | Animode 3 | Animode 4 |
|---------|-----------|-----------|-----------|-----------|-----------|
| Oven | door (revolute) | racks (prismatic) | all | - | - |
| Toilet | cover (revolute) | seat ring (revolute) | flush (prismatic) | all | - |
| Window | pane 1 (revolute) | pane 2 (revolute) | sliding (prismatic) | all revolute | all |
| Pot | lid lift (prismatic) | lid rotate (continuous) | URDF all | flip in-place | flip+place beside |
| Lamp | arm height (prismatic) | bulb slide (prismatic) | arm rotate (revolute) | all | - |
| BarChair | height (prismatic) | spin (continuous) | all | - | - |
| TV | tilt (revolute) | height (prismatic) | all | - | - |
| Pan | lid lift (prismatic) | - | - | - | - |

Joint selectors support multiple formats:
- `("type",)` — all significant joints of that type
- `("type", ordinal)` — nth joint by kinematic depth (0=shallowest, -1=deepest)
- `("type", "axis", "x"|"y"|"z")` — joints with given primary axis
- `("type", "sign", "+"|"-")` — joints filtered by limit sign

### Camera Views (32 total)

**16 fixed hemisphere views** (`hemi_00` to `hemi_15`):
- 4×4 grid on front hemisphere (azimuth ±67.5°, elevation 5°/25°/45°/65°)

**8 orbit views** (`orbit_00` to `orbit_07`):
- Camera travels ~180° from back to front of object
- Pairs via +Y/-Y sides, with constant/ascending/descending elevation

**8 sweep views** (`sweep_00` to `sweep_07`):
- Camera moves within front hemisphere
- Horizontal pans, vertical tilts, and diagonal paths

### Output Structure

```
outputs/motion_videos/{Factory}/{seed}/
  hemi_00_nobg.mp4          # animode 0, fixed view 0
  hemi_00_anim1_nobg.mp4    # animode 1, fixed view 0
  orbit_00_nobg.mp4         # animode 0, orbit view 0
  sweep_03_anim2_nobg.mp4   # animode 2, sweep view 3
  ...
```

Each video: 512×512, 30fps, 4 seconds (120 frames), transparent background (nobg).

## Key Scripts

| Script | Description |
|--------|-------------|
| `render_articulation.py` | Core Blender rendering script. URDF parsing, joint animation, multi-view camera, compositor setup |
| `batch_generate_all.py` | Batch pipeline: generate assets + render across multiple GPUs |
| `validate_animodes.py` | Validate animode joint selection across all factories and seeds |
| `split_and_visualize.py` | 2-part decomposition for part-aware training data |

## Credits

Built on [Infinite-Mobility](https://github.com/OpenRobotLab/Infinite-Mobility) by OpenRobotLab.
