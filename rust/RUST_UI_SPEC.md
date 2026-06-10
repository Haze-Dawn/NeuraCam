# NeuraCam Rust UI Suite — Technical Specification

## Overview

The NeuraCam Rust UI suite provides two complementary interfaces for the
NeuraCam face-tracking webcam system:

- **neuracam-tui** — A real-time terminal dashboard (ratatui) with retro-
  futuristic cyberpunk aesthetics, sparklines, gauges, and live metrics.
- **neuracam-gui** — A native GNOME desktop window (Adwaita/GTK4) for
  viewing the camera feed with OSD overlay and status bar.

Both connect to the Python backend via Unix domain sockets using a shared
length-prefixed binary protocol.

---

## 1. IPC Protocol

### 1.1 Socket Layout

| Socket | Path | Direction | Purpose |
|--------|------|-----------|---------|
| State | `/tmp/neuracam.sock` | Python → Rust | State JSON + JPEG frames |
| Input | `/tmp/neuracam_input.sock` | Rust → Python | Keyboard events |

### 1.2 Message Format

Every message on the state socket is length-prefixed with a type tag:

```
Offset  Size  Field
0       4     payload_length (big-endian u32)
4       1     message_type (u8: 0=state, 1=frame, 2=key)
5       N     payload (N = payload_length)
```

### 1.3 Message Types

| Type | Value | Direction | Payload | Frequency |
|------|-------|-----------|---------|-----------|
| `MSG_STATE` | `0x00` | Python→Rust | UTF-8 JSON | Every frame |
| `MSG_FRAME` | `0x01` | Python→Rust | JPEG bytes | Every frame |
| `MSG_KEY` | `0x02` | Rust→Python | Single ASCII byte | On user input |

### 1.4 State JSON Schema

```jsonc
{
  // System
  "fps": 29.5,
  "mode": "TRACKING",          // IDLE|TRACKING|TRACKING_HAND|LOCKED|HOME|SEARCH
  "tracking_target": "FACE",   // FACE|HAND
  "frame_w": 1280,
  "frame_h": 720,
  "timestamp": 1234567890.123,

  // Face Detection
  "face_detected": true,
  "face_x": 640.0,             // center X (pixels)
  "face_y": 360.0,             // center Y
  "face_w": 200.0,             // width
  "face_h": 250.0,             // height
  "face_confidence": 0.852,

  // Kalman Filter
  "kalman_uncertainty": 0.15,

  // PID Controller
  "pid_pan_error": 0.030,
  "pid_tilt_error": -0.020,
  "pid_pan_output": 2.5,
  "pid_tilt_output": -1.8,
  "pid_pan_p": 0.50,
  "pid_pan_i": 0.10,
  "pid_pan_d": -0.20,
  "pid_tilt_p": -0.30,
  "pid_tilt_i": 0.05,
  "pid_tilt_d": 0.10,

  // Gimbal / Servo
  "pan_angle": 90,
  "tilt_angle": 85,
  "pan_target": 92,
  "tilt_target": 84,

  // IMU
  "imu_pitch": 0.5,
  "imu_roll": -1.2,
  "imu_yaw": 0.3,

  // Hand / Gesture
  "hand_detected": false,
  "gesture": "NONE",
  "gesture_method": "",        // svm|rule

  // Controls
  "zoom_level": 1.0,
  "recording": false,
  "serial_connected": true,

  // Latency Profile (ms per component)
  "latency_ms": {
    "capture": 5.2, "detect": 15.1, "track": 0.3,
    "pid": 0.1, "display": 2.1, "gesture": 8.5, "ipc": 0.5
  },

  // Events
  "events": [
    "12:34:56 Mode: IDLE → TRACKING",
    "12:34:57 Face acquired (conf: 0.852)"
  ]
}
```

### 1.5 Input Protocol

Single-byte ASCII messages on the input socket:

| Byte | Action |
|------|--------|
| `q` | Quit / shutdown |
| `h` | Home gimbal |
| ` ` | Toggle lock tracking |
| `r` | Toggle recording |

---

## 2. neuracam-tui — Terminal Dashboard

### 2.1 Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Terminal UI | ratatui | 0.29 |
| Terminal backend | crossterm | 0.28 |
| JSON parsing | serde + serde_json | 1.x |
| Error handling | anyhow | 1.x |
| IPC | std::os::unix | stdlib |

### 2.2 Color Palette

The TUI uses a custom cyberpunk color scheme defined as RGB constants:

```
BG      (6, 6, 14)      Near-black with blue tint (background)
PANEL   (10, 10, 22)    Slightly lighter panel background
BDIM    (55, 62, 90)    Muted blue-gray (borders, separators)
TEXT    (185, 195, 215) Light blue-gray (body text)
TDIM    (75, 85, 115)   Dimmed text (labels, hints)
CYAN    (0, 215, 255)   Primary accent (headers, pan PID)
MAG     (200, 40, 255)  Magenta (section titles, tilt PID)
GRN     (0, 235, 75)    Green (good status, FPS, tracking)
YLW     (255, 185, 0)   Yellow (warnings)
ORG     (255, 100, 0)   Orange (search mode, hand tracking)
RED     (255, 35, 55)   Red (errors, idle, disconnection)
PNK     (255, 80, 160)  Pink (latency/events section titles)
```

### 2.3 Section Layout

The TUI uses a pure vertical stack layout optimized for narrow terminals
(72×22 minimum, 80×24 recommended). Section heights scale with terminal
size:

```
 ┌─ 1: Header ──────────────────────────────────────────── 1 line
 │ ◆NeuraCam ┃ TRACKING(FACE) ┃ 29.5fps ┃●REC SRL FCE HND Z1.3x 00:02:31
 │ ────────────────────────────────────────────────────────
 ├─ 2: Face ─────────────────── 20% of available, min 4, max 8 lines
 │ ┌──────────┐  BBox( 320, 240 120×150)  0.852
 │ │    ◉     │  Kalm ▓▓▓▓▓▓░░  0.15  Zoom ▓▓░░  1.3x
 │ └──────────┘  Frm #1,423  TRACKING  45f  Size 35.2%  Edge 50%
 ├─ 3: PID ──────────────────── 42% of available, 3 sparklines stacked
 │ Pan ┤▁▂▃▄▅▆▇████████▇▆▅▄▃▂├──  P+0.50  I+0.10  D-0.20
 │      Err+0.030  Out+2.5°
 │ Tilt┤████████▇▆▅▄▃▂▁▁▂▃▄▅▆├──  P-0.30  I+0.05  D+0.10
 │      Err-0.020  Out-1.8°
 │ FPS ┤████████████████████████████████├──  29.5 a28.7  Y:N Rec:○
 ├─ 4: System ────────────────── remaining space
 │   Gimbal P 90° ▓▓▓▓▓▓▓▓░░  T 85° ▓▓▓▓▓▓░░  92/84°
 │   IMU    P+0.5°  R-1.2°  Y+0.3°  Edge 50%
 │   Gest OPEN_PALM(svm)  Kalm ▓▓▓▓▓▓░░  0.15
 │   Hand ●DETECTED  Serial ●CON  Rec ○OFF
 ├─ 5: Latency ──────────────── 2 lines (title + data)
 │ Tot 23.3ms  det15.0  cap5.2  gst8.5  dis2.1  ipc0.5  trk0.3  pid0.1
 ├─ 6: Events ───────────────── 4 lines (title + 3 events)
 │ [12:34:56] Mode: IDLE → TRACKING
 │ [12:34:57] Face acquired (conf: 0.852)
 └─ 7: Controls ─────────────── 2 lines (title + shortcuts)
   q:Quit   h:Home   Space:Lock   r:Rec
```

### 2.4 Widget Reference

#### 2.4.1 Header
- **Type**: `Paragraph`
- **Content**: Single `Line` of `Span` elements
- **Style**: CYAN bold for title, mode color with background tint for mode tag
- **Data**: Mode, tracking target, FPS, REC indicator, SRL/FACE/HAND status dots,
  zoom level, uptime

#### 2.4.2 Face Panel
- **Structure**: Title line (`─ Face ──`), then horizontal split into minimap (35%)
  and info text (65%)
- **Minimap**: ASCII border with `┌┐└┘│─` characters, `◉` dot at face position,
  color-coded by Kalman uncertainty (green < 0.3, yellow < 0.6, red > 0.6)
- **Info**: BBox coordinates, confidence, Kalman gauge (`▓`/`░`), zoom gauge,
  frame counter, mode, mode-frame counter, face size ratio, edge proximity

#### 2.4.3 PID Panel
- **Structure**: Title line (`─ PID ──`), then 3 sparkline rows (Pan, Tilt, FPS)
- **Sparkline**: `Sparkline` widget with `Borders::ALL`, colored per-axis
  (cyan=pan, magenta=tilt, green=FPS)
- **Labels**: P/I/D terms on first line, error + output on second line
- **FPS row**: Current fps, rolling average, face/hand/rec status

#### 2.4.4 System Panel
- **Structure**: Title line (`─ System ──`), split into Gimbal + Detection
- **Gimbal**: Pan/Tilt angles with `▓`/`░` gauges, target angles, IMU P/R/Y,
  edge proximity percentage
- **Detection**: Gesture name + method, Kalman gauge, Hand/Serial/Rec status

#### 2.4.5 Latency Bar
- **Type**: `Paragraph`
- **Content**: Total latency (bold, color-coded) followed by per-component
  latencies sorted by value (detect, capture, gesture, display, ipc, track, pid)
- **Color**: Green < 5ms, Yellow < 15ms, Red > 15ms

#### 2.4.6 Events
- **Type**: `List` with `Borders::TOP`
- **Content**: Last 3 events, each with gray timestamp and white message body

#### 2.4.7 Controls
- **Type**: `Paragraph`
- **Content**: Keyboard shortcut hints with CYAN bold keys and gray descriptions

### 2.5 Gauge Rendering

Custom `gauge()` function renders filled/empty using Unicode block characters:

```rust
fn gauge(pct: u16, w: usize) -> String {
    let f = (pct as usize * w / 100).min(w);
    let e = w.saturating_sub(f);
    format!("{}{}", "▓".repeat(f), "░".repeat(e))
}
```

- `▓` (U+2593, dark shade) for filled portion
- `░` (U+2591, light shade) for empty portion
- Width `w` is typically 8 or 10 characters

### 2.6 Derived Metrics

The TUI computes additional metrics not directly in the IPC state:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Face size ratio | `(face_w×face_h) / (frame_w×frame_h) × 100` | % of frame the face occupies |
| Edge proximity | `min(pan-lo, hi-pan) / (hi-lo) × 100` | Distance to gimbal travel limits |
| Mode-frame counter | Incremented each frame in same mode | Frames since last mode change |
| FPS rolling average | `∑fps_history / len × 30/100` | Average FPS over sparkline window |
| PID error normalization | `abs(error) × 150` clamped to 0-100 | Sparkline data range |

### 2.7 Thread Model

```
┌─────────────────────────────────────────┐
│             Main Thread                  │
│  ratatui event loop (~30fps)            │
│  1. Check channel for new state         │
│  2. Update sparkline history buffers    │
│  3. Render frame via ratatui            │
│  4. Handle keyboard input               │
│  5. Send key events to input socket     │
└──────────────┬──────────────────────────┘
               │ mpsc::channel
┌──────────────▼──────────────────────────┐
│           Reader Thread                  │
│  Blocks on UnixStream.read_exact()      │
│  1. Read 5-byte header                  │
│  2. Read payload                        │
│  3. Type 0 → deserialize JSON → send    │
│  4. Type 1 → discard (TUI ignores JPEG) │
│  5. On error → reconnect after 2s       │
└─────────────────────────────────────────┘
```

### 2.8 History Buffers

| Buffer | Size | Update | Display |
|--------|------|--------|---------|
| `bufs.pan` | 40 | Every frame | Pan error sparkline |
| `bufs.tilt` | 40 | Every frame | Tilt error sparkline |
| `bufs.fps` | 40 | Every frame | FPS sparkline |

### 2.9 Edge Cases

| Condition | Behavior |
|-----------|----------|
| No Python process | Reader thread retries every 2s, header shows `○` disconnected |
| Connection lost >3s | `connected = false`, OSD shows disconnected state |
| Terminal <72×22 | Error message: "Need 72×22 min" |
| No face detected | Minimap shows `⛔ NO FACE` in RED bold |
| Gesture NONE | Dimmed gray text |
| Zero FPS | RED display, flatlined sparkline |
| Serial disconnected | RED indicator in header and system panel |

---

## 3. neuracam-gui — Native GNOME Window

### 3.1 Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Application framework | libadwaita (adw) | 0.9 |
| GUI toolkit | gtk4 | 0.11 |
| Image loading | gdk-pixbuf | 0.22 |
| JSON parsing | serde + serde_json | 1.x |
| IPC | std::os::unix | stdlib |

### 3.2 Widget Tree

```
AdwApplicationWindow
└── GtkBox (vertical)
    ├── AdwHeaderBar
    │   ├── [title] GtkLabel "NeuraCam"
    │   └── [end] GtkBox (horizontal)
    │       ├── GtkButton btn_home   [go-home-symbolic]
    │       ├── GtkToggleButton btn_lock [channel-insecure-secure]
    │       ├── GtkToggleButton btn_rec  [media-record-symbolic]
    │       └── GtkMenuButton menu_btn [open-menu-symbolic]
    │           └── GtkPopover
    │               └── GtkBox (vertical)
    │                   ├── GtkCheckButton "Show Overlay"
    │                   ├── GtkCheckButton "Show Status Bar"
    │                   ├── GtkSeparator
    │                   ├── GtkLabel "O: overlay  S: status bar"
    │                   ├── GtkSeparator
    │                   └── GtkButton "Quit"
    ├── GtkOverlay (vexpand=true, hexpand=true)
    │   ├── GtkPicture (camera feed, aspect-ratio locked)
    │   └── [overlay] GtkRevealer (crossfade, 200ms)
    │       └── GtkBox (vertical)
    │           ├── GtkLabel (mode, osd-mode style)
    │           ├── GtkLabel (FPS, osd-label style)
    │           ├── GtkLabel (REC, osd-label style)
    │           └── GtkLabel (warning, osd-label style, bottom-right)
    └── GtkRevealer (slide-up, 200ms)
        └── GtkBox (horizontal, spacing=6)
            ├── GtkLabel mode_st
            ├── GtkSeparator (vertical)
            ├── GtkLabel fps_st
            ├── GtkSeparator (vertical)
            ├── GtkLabel rec_st
            ├── GtkSeparator (vertical)
            ├── GtkLabel srl_st
            ├── GtkSeparator (vertical)
            ├── GtkLabel zoom_st
            ├── GtkSeparator (vertical)
            ├── GtkLabel "Kalman"
            ├── GtkLevelBar (kalman uncertainty, 0-1)
            └── GtkLabel gest_st
```

### 3.3 Window Properties

| Property | Value |
|----------|-------|
| Application ID | `com.neuracam.viewer` |
| Default size | 960 × 640 |
| Min size | 400 × 300 |
| Title | `NeuraCam` |
| Background | `#0a0a14` |
| CSS provider | Priority `APPLICATION` |

### 3.4 CSS Styles

```css
.osd-label {
    font-size: 15px; font-weight: bold; color: #ffffff;
    background: rgba(0, 0, 0, 0.55);
    padding: 4px 10px; border-radius: 6px;
}
.osd-mode { font-size: 18px; }
.feed-area { background: #0a0a14; }
window.background { background: #0a0a14; }
.toolbar-btn {
    min-width: 32px; min-height: 32px;
    border-radius: 6px; margin: 2px;
}
.toolbar-btn:hover { background: rgba(255,255,255,0.12); }
.toolbar-btn:active { background: rgba(255,255,255,0.20); }
.toolbar-btn:checked {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.3);
}
```

### 3.5 OSD Overlay

The OSD (on-screen display) overlays status information on the camera feed:

| Field | Position | Content | Color |
|-------|----------|---------|-------|
| Mode | Top-left, y=12 | `[ TRACKING (FACE) ]` | Per-mode (green/orange/yellow/red) |
| FPS | Top-left, y=46 | `29.5 fps` | Green > 20, Yellow > 10, Red |
| REC | Top-left, y=76 | `● REC` | Red, hidden when not recording |
| Warning | Bottom-right | `⚠ Serial disconnected` | Orange, hidden when OK |

The entire OSD box is wrapped in a `GtkRevealer` with `Crossfade` transition
(200ms), controllable via:
- Menu: `☰ → Show Overlay` checkbox
- Keyboard: `O` key toggles

### 3.6 Status Bar

The bottom status bar is a horizontal `GtkBox` with native GTK4 widgets:

| Widget | Content | Format |
|--------|---------|--------|
| `Label` mode_st | Mode + target | Color-coded, bold |
| `Separator` | Vertical line | |
| `Label` fps_st | `29.5 fps` | Plain text |
| `Separator` | | |
| `Label` rec_st | `● REC` or `○ REC` | Red bold when recording |
| `Separator` | | |
| `Label` srl_st | `● SRL` or `○ SRL` | Green/Red |
| `Separator` | | |
| `Label` zoom_st | `Z 1.3x` | Visible only when zoom > 1.0 |
| `Separator` | | |
| `Label` | `Kalman` | Static label |
| `LevelBar` | 0.0–1.0 continuous | Width 80px, tooltip shows value |
| `Label` gest_st | `Gesture: OPEN_PALM` | Visible only when gesture active |

The status bar is wrapped in a `GtkRevealer` with `SlideUp` transition (200ms):
- Menu: `☰ → Show Status Bar` checkbox
- Keyboard: `S` key toggles

### 3.7 Toolbar Buttons

| Button | Type | Icon | Action | State |
|--------|------|------|--------|-------|
| Home | `GtkButton` | `go-home-symbolic` | Sends `h` key | Momentary |
| Lock | `GtkToggleButton` | `channel-insecure/secure-symbolic` | Sends ` ` (space) | Toggle + icon swap |
| Record | `GtkToggleButton` | `media-record-symbolic` | Sends `r` key | Toggle |
| Menu | `GtkMenuButton` | `open-menu-symbolic` | Opens popover | Shows/hides |

All toolbar buttons use the `.toolbar-btn` CSS class for hover/press/checked
visual feedback.

### 3.8 Thread Model

```
┌──────────────────────────────────────────┐
│          Main Thread (GTK event loop)     │
│  gtk4::glib::timeout_add_local (33ms)    │
│  1. Check mpsc::Receiver for new frames  │
│  2. Decode JPEG → Pixbuf → Picture       │
│  3. Update OSD labels from state         │
│  4. Update status bar widgets from state │
│  5. Sync Revealer visibility with toggles│
│  GtkEventControllerKey → send_key()      │
└──────────────┬───────────────────────────┘
               │ mpsc::channel
┌──────────────▼───────────────────────────┐
│          Reader Thread                    │
│  Blocks on UnixStream.read_exact()       │
│  1. Read 5-byte header                   │
│  2. Read payload                         │
│  3. Buffer MSG_STATE and MSG_FRAME       │
│  4. When both available → pair + send    │
│  5. On error → reconnect after 2s        │
└──────────────────────────────────────────┘
```

### 3.9 Keyboard Shortcuts

| Key | Action | Scope |
|-----|--------|-------|
| `Q` | Send quit signal to server | Application |
| `H` | Home gimbal | Application |
| `Space` | Toggle lock/unlock | Application |
| `R` | Toggle recording | Application |
| `O` | Toggle OSD overlay visibility | GUI only |
| `S` | Toggle status bar visibility | GUI only |

Key events are captured by `GtkEventControllerKey` attached to the window.
The `O` and `S` keys are local GUI-only toggles that do not send IPC messages.

### 3.10 Icon Names

| Logical name | Icon theme name | Source |
|-------------|-----------------|--------|
| Home | `go-home-symbolic` | Adwaita actions |
| Lock (active) | `channel-secure-symbolic` | Adwaita status |
| Lock (inactive) | `channel-insecure-symbolic` | Adwaita status |
| Record | `media-record-symbolic` | Adwaita media |
| Menu | `open-menu-symbolic` | Adwaita actions |
| Quit | `application-exit-symbolic` | Adwaita actions |

### 3.11 Edge Cases

| Condition | Behavior |
|-----------|----------|
| No Python process | Reader thread retries every 2s, OSD shows `[ DISCONNECTED ]` in RED |
| Connection lost | Same as above, picture shows last received frame |
| JPEG decode failure | Frame silently dropped, previous frame retained |
| IPC channel disconnect | OSD shows `[ IPC ERROR ]`, update loop stops |
| Terminal (no DISPLAY) | GUI won't launch (GTK requires display server) |
| Window minimized | GTK still processes updates but doesn't render |

---

## 4. Shared Crate

### 4.1 `neuracam-shared` — Protocol Types

**Path**: `rust/shared/src/lib.rs`

```rust
pub const MSG_STATE: u8 = 0;
pub const MSG_FRAME: u8 = 1;
pub const MSG_KEY: u8 = 2;

pub struct NeuraCamState {
    // 40+ fields matching the JSON schema above
    // All fields are pub with serde::Deserialize
}

pub fn read_msg(stream: &mut impl Read) -> io::Result<(u8, Vec<u8>)>
```

The `read_msg` function reads a single message from any `Read` source:
1. Read 5-byte header (4 bytes length big-endian + 1 byte type)
2. Read `length` bytes of payload
3. Return `(msg_type, payload_bytes)`

---

## 5. Building and Running

### 5.1 Prerequisites

```bash
# Arch Linux
sudo pacman -S gtk4 libadwaita

# Debian/Ubuntu
sudo apt install libgtk-4-dev libadwaita-1-dev

# Rust toolchain
rustup default stable
```

### 5.2 Build

```bash
cd rust

# Build everything
cargo build --release

# Build individual
cargo build --release -p neuracam-tui
cargo build --release -p neuracam-gui
```

### 5.3 Run

```bash
# Terminal 1: Start the Python backend (or mock)
cd /path/to/NeuraCam\ Repo
python3 scripts/mock_neuracam.py

# Terminal 2: Launch TUI dashboard
/path/to/NeuraCam\ Repo/rust/target/release/neuracam-tui

# Terminal 3: Launch GNOME GUI window
/path/to/NeuraCam\ Repo/rust/target/release/neuracam-gui
```

### 5.4 Binary Sizes

| Binary | Size (release) |
|--------|----------------|
| `neuracam-tui` | 1.3 MB |
| `neuracam-gui` | 972 KB |

---

## 6. File Structure

```
rust/
├── Cargo.toml                    # Workspace definition
├── RUST_UI_SPEC.md               # This document
├── shared/
│   ├── Cargo.toml
│   └── src/lib.rs                # NeuraCamState, read_msg, constants
├── tui/
│   ├── Cargo.toml
│   └── src/main.rs               # App struct, render, IPC reader
└── gui/
    ├── Cargo.toml
    └── src/main.rs               # Adwaita app, widgets, IPC reader
```

---

## 7. Comparison: TUI vs GUI

| Aspect | TUI (ratatui) | GUI (Adwaita/GTK4) |
|--------|---------------|-------------------|
| Primary purpose | Real-time dashboard monitoring | Camera feed viewing |
| Visual style | Retro-futuristic cyberpunk | Native GNOME/Adwaita |
| Layout | Vertical stack, narrow optimized | Desktop window with header/content/status |
| Camera feed | ASCII minimap only | Full JPEG frame display |
| Sparklines | 3 rolling graphs (pan/tilt/fps) | LevelBar for Kalman only |
| Data density | Very high (all metrics on screen) | Moderate (key metrics) |
| Terminal required | Yes (ALT screen) | No |
| Display server | No (any terminal) | Yes (X11/Wayland) |
| Togglable overlays | N/A | OSD + Status Bar (O/S keys) |
| Controls | Keyboard only | Toolbar buttons + keyboard |
| Input feedback | N/A | CSS hover/press/checked states |
| Dependencies | ratatui, crossterm | gtk4, libadwaita, gdk-pixbuf |
| Binary size | 1.3 MB | 972 KB |
