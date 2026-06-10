use std::collections::VecDeque;
use std::io::{self, Write};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Terminal;

use neuracam_shared::{read_msg, NeuraCamState, MSG_STATE};

const SOCKET_PATH: &str = "/tmp/neuracam.sock";
const INPUT_SOCK_PATH: &str = "/tmp/neuracam_input.sock";
const SPARK_WIN: usize = 35;
const EVT_SHOW: usize = 6;
const RECONNECT_DELAY: Duration = Duration::from_secs(2);

// ── TERMINAL PALETTE: WARM GREY ON DARK ──
const BG: Color = Color::Rgb(0x0D, 0x0D, 0x0D);
const TXT: Color = Color::Rgb(0xCC, 0xCC, 0xCC);
const DTXT: Color = Color::Rgb(0x66, 0x66, 0x66);
const BTXT: Color = Color::Rgb(0x44, 0x44, 0x44);
const RED: Color = Color::Rgb(0xE0, 0x55, 0x55);
const YLW: Color = Color::Rgb(0xD4, 0xA0, 0x40);
const ACC: Color = Color::Rgb(0x80, 0xAA, 0xCC);


fn mode_color(m: &str) -> Color {
    match m {
        "TRACKING" => TXT,
        "TRACKING_HAND" => YLW,
        "LOCKED" => YLW,
        "IDLE" => DTXT,
        "SEARCH" => YLW,
        "HOME" => ACC,
        _ => TXT,
    }
}

fn uptime_str(d: Duration) -> String {
    let s = d.as_secs();
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let sec = s % 60;
    if h > 0 {
        format!("{:02}:{:02}:{:02}", h, m, sec)
    } else {
        format!("{:02}:{:02}", m, sec)
    }
}

fn face_ratio(s: &NeuraCamState) -> f64 {
    if s.face_detected && s.frame_w > 0.0 && s.frame_h > 0.0 {
        (s.face_w * s.face_h) / (s.frame_w * s.frame_h) * 100.0
    } else {
        0.0
    }
}

fn gauge(pct: u16, w: usize) -> String {
    let f = (pct as usize * w / 100).min(w);
    let e = w.saturating_sub(f);
    format!("{}{}", "▓".repeat(f), "░".repeat(e))
}

fn edge_dist(a: i64, lo: i64, hi: i64) -> u16 {
    let r = hi - lo;
    if r == 0 {
        return 0;
    }
    let d = (a - lo).min(hi - a).max(0) as f64;
    (d / r as f64 * 100.0) as u16
}


fn fmt_pid(v: f64) -> String {
    if v.abs() < 0.001 {
        format!("{:+.2E}", v)
    } else if v.abs() < 0.01 {
        format!("{:+.4}", v)
    } else if v.abs() < 1.0 {
        format!("{:+.3}", v)
    } else {
        format!("{:+.2}", v)
    }
}

fn status_bool(v: bool, yes: &'static str, no: &'static str) -> &'static str {
    if v { yes } else { no }
}

// ── BUFFERS ──
struct Bufs {
    pan: VecDeque<u64>,
    tilt: VecDeque<u64>,
    fps: VecDeque<u64>,
    pan_err: VecDeque<u64>,
    tilt_err: VecDeque<u64>,
}

impl Bufs {
    fn new() -> Self {
        Self {
            pan: VecDeque::with_capacity(SPARK_WIN),
            tilt: VecDeque::with_capacity(SPARK_WIN),
            fps: VecDeque::with_capacity(SPARK_WIN),
            pan_err: VecDeque::with_capacity(SPARK_WIN),
            tilt_err: VecDeque::with_capacity(SPARK_WIN),
        }
    }
    fn push(&mut self, s: &NeuraCamState) {
        let entries: [(&mut VecDeque<u64>, u64); 5] = [
            (
                &mut self.pan,
                (s.pid_pan_error.abs() * 200.0).min(100.0) as u64,
            ),
            (
                &mut self.tilt,
                (s.pid_tilt_error.abs() * 200.0).min(100.0) as u64,
            ),
            (
                &mut self.fps,
                (s.fps * 100.0 / 30.0).min(100.0) as u64,
            ),
            (
                &mut self.pan_err,
                (s.pid_pan_error.abs() * 200.0).min(100.0) as u64,
            ),
            (
                &mut self.tilt_err,
                (s.pid_tilt_error.abs() * 200.0).min(100.0) as u64,
            ),
        ];
        for (b, v) in entries {
            b.push_back(v);
            if b.len() > SPARK_WIN {
                b.pop_front();
            }
        }
    }
    fn avg_fps(&self) -> f64 {
        if self.fps.is_empty() {
            return 0.0;
        }
        self.fps.iter().sum::<u64>() as f64 / self.fps.len() as f64 * 30.0 / 100.0
    }
}

// ── APP ──
struct App {
    s: NeuraCamState,
    connected: bool,
    bufs: Bufs,
    rx: Receiver<NeuraCamState>,
    t0: Instant,
    last_ts: Instant,
    inp: Option<UnixStream>,
    frames: u64,
    prev_mode: String,
    mode_frame: u64,
}

impl App {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || loop {
            match UnixStream::connect(SOCKET_PATH) {
                Ok(mut st) => loop {
                    match read_msg(&mut st) {
                        Ok((t, p)) => {
                            if t == MSG_STATE {
                                if let Ok(st) = serde_json::from_slice::<NeuraCamState>(&p) {
                                    let _ = tx.send(st);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                },
                Err(_) => {}
            }
            thread::sleep(RECONNECT_DELAY);
        });
        Self {
            s: NeuraCamState::default(),
            connected: false,
            bufs: Bufs::new(),
            rx,
            t0: Instant::now(),
            last_ts: Instant::now(),
            inp: None,
            frames: 0,
            prev_mode: String::new(),
            mode_frame: 0,
        }
    }

    fn connect_input(&mut self) {
        if self.inp.is_some() {
            return;
        }
        if let Ok(s) = UnixStream::connect(INPUT_SOCK_PATH) {
            let _ = s.set_nonblocking(true);
            self.inp = Some(s);
        }
    }

    fn send_key(&mut self, k: char) {
        if let Some(ref mut s) = self.inp {
            let _ = s.write_all(&[k as u8]);
        }
    }

    fn tick(&mut self) {
        self.connect_input();
        let now = Instant::now();
        while let Ok(st) = self.rx.try_recv() {
            self.s = st;
            self.connected = true;
            self.frames += 1;
            self.bufs.push(&self.s);
            self.last_ts = now;
            if self.s.mode != self.prev_mode {
                self.prev_mode.clone_from(&self.s.mode);
                self.mode_frame = 0;
            }
            self.mode_frame += 1;
        }
        if now.duration_since(self.last_ts) > Duration::from_secs(3) {
            self.connected = false;
        }
    }

    fn handle_input(&mut self, t: Duration) -> Result<bool> {
        if event::poll(t)? {
            if let Event::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char('q') => {
                        self.send_key('q');
                        return Ok(false);
                    }
                    KeyCode::Char('h') => self.send_key('h'),
                    KeyCode::Char(' ') => self.send_key(' '),
                    KeyCode::Char('r') => self.send_key('r'),
                    _ => {}
                }
            }
        }
        Ok(true)
    }

    fn run(
        &mut self,
        term: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Result<()> {
        loop {
            self.tick();
            term.draw(|f| self.render(f))?;
            if !self.handle_input(Duration::from_millis(33))? {
                break;
            }
        }
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════
    // RENDER
    // ═══════════════════════════════════════════════════════════════

    fn render(&self, f: &mut ratatui::Frame) {
        let a = f.area();
        if a.width < 90 || a.height < 28 {
            f.render_widget(
                Paragraph::new("MIN TERMINAL: 90X28")
                    .alignment(Alignment::Center)
                    .style(Style::default().fg(RED)),
                a,
            );
            return;
        }

        let buf = f.buffer_mut();
        for y in a.top()..a.bottom() {
            for x in a.left()..a.right() {
                buf[(x, y)].set_bg(BG);
            }
        }

        // ── LAYOUT ──
        // [hdr 1] [sep 1] [top N] [sep 1] [mid N] [sep 1] [bot N] [sep 1] [ftr 2]

        let avail = a.height.saturating_sub(6); // 5 separators + 1 header = lines eaten
        let top_h = (avail * 40 / 100).max(12).min(18);
        let mid_h = (avail * 30 / 100).max(6).min(12);
        let bot_h = avail.saturating_sub(top_h + mid_h).max(4).min(7);

        let [hdr, s1, top, s2, mid, s3, bot, s4, ftr] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(top_h),
            Constraint::Length(1),
            Constraint::Length(mid_h),
            Constraint::Length(1),
            Constraint::Length(bot_h),
            Constraint::Length(1),
            Constraint::Length(2),
        ])
        .areas(a);

        // Top: health(28) | face_local(44) | state(28)
        let [top_l, top_c, top_r] = Layout::horizontal([
            Constraint::Ratio(28, 100),
            Constraint::Ratio(44, 100),
            Constraint::Ratio(28, 100),
        ])
        .areas(top);

        // Mid: axis_ctrl(28) | signal(44) | events(28)
        let [mid_l, mid_c, mid_r] = Layout::horizontal([
            Constraint::Ratio(28, 100),
            Constraint::Ratio(44, 100),
            Constraint::Ratio(28, 100),
        ])
        .areas(mid);

        // Bot: sig_lvls(22) | actuator(78)
        let [bot_l, bot_r] =
            Layout::horizontal([Constraint::Ratio(22, 100), Constraint::Ratio(78, 100)])
                .areas(bot);

        // ── DRAW GRID LINES (VERTICAL) - only through top+mid sections ──
        let sep_x = [
            top_l.right(),
            top_c.right(),
        ];
        for &sx in &sep_x {
            for y in top.y..s3.y {
                buf[(sx.saturating_sub(1), y)].set_symbol("│");
                buf[(sx.saturating_sub(1), y)].set_fg(BTXT);
            }
        }

        self.render_header(f, hdr);
        self.render_separator(f, s1);
        self.render_system_health(f, top_l);
        self.render_face_localization(f, top_c);
        self.render_system_state(f, top_r);
        self.render_separator(f, s2);
        self.render_axis_control(f, mid_l);
        self.render_signal_analysis(f, mid_c);
        self.render_event_log(f, mid_r);
        self.render_separator(f, s3);
        self.render_signal_levels(f, bot_l);
        self.render_actuator_state(f, bot_r);
        self.render_separator(f, s4);
        self.render_footer(f, ftr);
    }

    fn render_separator(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("═");
            buf[(x, a.y)].set_fg(DTXT);
        }
    }

    // ── HEADER ──
    fn render_header(&self, f: &mut ratatui::Frame, a: Rect) {
        let m = &self.s.mode;
        let mt = if m == "IDLE" || m == "SEARCH" {
            format!("[{}]", m)
        } else {
            format!("{}({})", m, self.s.tracking_target)
        };
        let upt = uptime_str(self.t0.elapsed());

        let fpc = if self.s.fps > 20.0 {
            TXT
        } else if self.s.fps > 10.0 {
            YLW
        } else {
            RED
        };

        let line = Line::from(vec![
            Span::styled(
                " ◆NEURACAM V3.14 ",
                Style::default().fg(TXT).add_modifier(Modifier::BOLD),
            ),
            Span::styled("│", Style::default().fg(BTXT)),
            Span::styled(
                format!(" MODE:{} ", mt),
                Style::default()
                    .fg(mode_color(m))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("│", Style::default().fg(BTXT)),
            Span::styled(
                format!(" FPS:{:.1} ", self.s.fps),
                Style::default().fg(fpc).add_modifier(Modifier::BOLD),
            ),
            Span::styled("│", Style::default().fg(BTXT)),
            Span::styled(
                format!(" {}CAM", status_bool(self.connected, "●", "○")),
                Style::default().fg(if self.connected { TXT } else { RED }),
            ),
            Span::styled(
                format!(" {}SRL", status_bool(self.s.serial_connected, "●", "○")),
                Style::default().fg(if self.s.serial_connected {
                    TXT
                } else {
                    RED
                }),
            ),
            Span::styled(
                format!(" {}FCE", status_bool(self.s.face_detected, "●", "○")),
                Style::default().fg(if self.s.face_detected {
                    TXT
                } else {
                    DTXT
                }),
            ),
            Span::styled(
                format!(" {}HND", status_bool(self.s.hand_detected, "●", "○")),
                Style::default().fg(if self.s.hand_detected {
                    TXT
                } else {
                    DTXT
                }),
            ),
            Span::styled(
                format!(" {}REC", status_bool(self.s.recording, "◆", "○")),
                Style::default().fg(if self.s.recording {
                    RED
                } else {
                    DTXT
                }),
            ),
            Span::styled("│", Style::default().fg(BTXT)),
            Span::styled(
                format!(" ZOOM:{:.1}X", self.s.zoom_level),
                Style::default().fg(YLW),
            ),
            Span::raw(" "),
            Span::styled(upt, Style::default().fg(DTXT)),
        ]);
        f.render_widget(Paragraph::new(line).style(Style::default().bg(BG)), a);
    }

    // ── SYSTEM HEALTH ──
    fn render_system_health(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        // Title bar
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }
        for (i, ch) in " SYSTEM HEALTH ".chars().enumerate() {
            let cx = a.x + 2 + i as u16;
            if cx < a.right() {
                buf[(cx, a.y)].set_symbol(&ch.to_string());
                buf[(cx, a.y)].set_fg(TXT);
                buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
            }
        }

        let inner = Rect::new(a.x, a.y + 1, a.width, a.height - 1);
        let dw = inner.width as usize;

        let cam_st = status_bool(self.connected, "OK", "DISCON");
        let ser_st = status_bool(self.s.serial_connected, "ONLINE", "OFFLINE");
        let pan_st = status_bool(self.connected, "READY", "LOST");
        let tilt_st = status_bool(self.connected, "READY", "LOST");
        let gesture_st = if self.s.gesture != "NONE" {
            &self.s.gesture
        } else {
            "N/A"
        };
        let kal_st = if self.s.kalman_uncertainty < 0.3 {
            "TRK"
        } else if self.s.kalman_uncertainty < 0.6 {
            "PRED"
        } else {
            "SEARCH"
        };

        let lat = &self.s.latency_ms;
        let det_ms = lat.as_ref().and_then(|m| m.get("detect")).copied();
        let ges_ms = lat.as_ref().and_then(|m| m.get("gesture")).copied();
        let track_ms = lat.as_ref().and_then(|m| m.get("track")).copied();
        let cap_ms = lat.as_ref().and_then(|m| m.get("capture")).copied();
        let pid_ms = lat.as_ref().and_then(|m| m.get("pid")).copied();

        let gest_method = if self.s.gesture_method.is_empty() {
            "N/A"
        } else {
            self.s.gesture_method.as_str()
        };

        let items: Vec<(&str, String, Color)> = vec![
            ("CAMERA", cam_st.into(), if cam_st == "OK" { TXT } else { RED }),
            ("SERIAL", ser_st.into(), if ser_st == "ONLINE" { TXT } else { RED }),
            ("IMU", "OK".into(), TXT),
            ("PAN", pan_st.into(), if pan_st == "READY" { TXT } else { RED }),
            ("TILT", tilt_st.into(), if tilt_st == "READY" { TXT } else { RED }),
            (
                "FACE DET",
                format!(
                    "{}MS  CNF:{:.3}",
                    det_ms.map(|m| format!("{:.1}", m)).unwrap_or("--".into()),
                    self.s.face_confidence
                ),
                TXT,
            ),
            (
                "GESTURE",
                format!(
                    "{} {}MS",
                    gest_method.to_uppercase(),
                    ges_ms.map(|m| format!("{:.1}", m)).unwrap_or("--".into()),
                ),
                if gesture_st == "N/A" { YLW } else { TXT },
            ),
            (
                "KALMAN",
                format!(
                    "{}  TRK:{}MS",
                    kal_st,
                    track_ms.map(|m| format!("{:.1}", m)).unwrap_or("--".into())
                ),
                if kal_st == "TRK" {
                    TXT
                } else if kal_st == "PRED" {
                    YLW
                } else {
                    RED
                },
            ),
            (
                "CAPTURE",
                cap_ms.map(|m| format!("{:.1}MS", m)).unwrap_or("--".into()),
                TXT,
            ),
            (
                "PID CTRL",
                pid_ms.map(|m| format!("{:.1}MS", m)).unwrap_or("--".into()),
                TXT,
            ),
        ];

        let mut lines = Vec::with_capacity(items.len());
        for (label, value, color) in &items {
            let vw = value.len();
            let dots = dw.saturating_sub(label.len() + vw + 4);
            let d = ".".repeat(dots);
            lines.push(Line::from(vec![
                Span::styled(format!("  {}", label), Style::default().fg(DTXT)),
                Span::styled(d, Style::default().fg(DTXT)),
                Span::styled(
                    format!(" {}", value),
                    Style::default()
                        .fg(*color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        }

        // Edge proximity
        if lines.len() as u16 + 1 < inner.height {
            let ep = edge_dist(self.s.pan_angle, 0, 180);
            let et = edge_dist(self.s.tilt_angle, 0, 180);
            lines.push(Line::from(vec![
                Span::styled("  EDGE LIMIT", Style::default().fg(DTXT)),
                Span::styled(
                    format!(
                        "{} P:{}% T:{}%",
                        ".".repeat(
                            dw.saturating_sub("EDGE LIMIT".len() + 13)
                        ),
                        ep,
                        et
                    ),
                    Style::default().fg(DTXT),
                ),
            ]));
        }

        // Face ratio
        if lines.len() as u16 + 1 < inner.height {
            let fr = face_ratio(&self.s);
            lines.push(Line::from(vec![
                Span::styled("  FACE RATIO", Style::default().fg(DTXT)),
                Span::styled(
                    format!(
                        "{} {:.1}%",
                        ".".repeat(dw.saturating_sub("FACE RATIO".len() + 7)),
                        fr
                    ),
                    Style::default().fg(TXT),
                ),
            ]));
        }

        f.render_widget(Paragraph::new(lines).style(Style::default().bg(BG)), inner);
    }

    // ── FACE LOCALIZATION ──
    fn render_face_localization(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }
        let title = format!(
            " FACE LOCALIZATION  {:>4}×{:<4}  F#{:<6} ",
            self.s.frame_w as u64,
            self.s.frame_h as u64,
            self.frames
        );
        for (i, ch) in title.chars().enumerate() {
            let cx = a.x + 2 + i as u16;
            if cx < a.right() {
                buf[(cx, a.y)].set_symbol(&ch.to_string());
                buf[(cx, a.y)].set_fg(TXT);
                buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
            }
        }

        let map_h = (a.height - 2).max(4);
        let info_h = (a.height - 1).saturating_sub(map_h);
        let map_area = Rect::new(a.x + 1, a.y + 1, a.width.saturating_sub(2), map_h);
        let info_area = Rect::new(a.x, a.y + 1 + map_h, a.width, info_h);

        if self.s.face_detected {
            let fw = self.s.frame_w.max(1.0);
            let fh = self.s.frame_h.max(1.0);
            let nx = (self.s.face_x / fw).clamp(0.0, 1.0);
            let ny = (self.s.face_y / fh).clamp(0.0, 1.0);
            let mw = map_area.width as f64;
            let mh = map_area.height as f64;

            let chx = map_area.x + (map_area.width / 2);
            let chy = map_area.y + (map_area.height / 2);

            let dot_x = map_area.x + (nx * mw) as u16;
            let dot_y = map_area.y + (ny * mh) as u16;

            // Crosshair lines
            for x in map_area.x..map_area.right() {
                if x == chx {
                    continue;
                }
                buf[(x, chy)].set_symbol("─");
                buf[(x, chy)].set_fg(DTXT);
                buf[(x, chy)].set_bg(BG);
            }
            for y in map_area.y..map_area.bottom() {
                if y == chy {
                    continue;
                }
                buf[(chx, y)].set_symbol("│");
                buf[(chx, y)].set_fg(DTXT);
                buf[(chx, y)].set_bg(BG);
            }

            if chx < map_area.right() && chy < map_area.bottom() {
                buf[(chx, chy)].set_symbol("┼");
                buf[(chx, chy)].set_fg(DTXT);
                buf[(chx, chy)].set_bg(BG);
            }

            // Map border
            for x in map_area.x..map_area.right() {
                buf[(x, map_area.y)].set_symbol("─");
                buf[(x, map_area.y)].set_fg(BTXT);
                buf[(x, map_area.y)].set_bg(BG);
                buf[(x, map_area.bottom() - 1)].set_symbol("─");
                buf[(x, map_area.bottom() - 1)].set_fg(BTXT);
                buf[(x, map_area.bottom() - 1)].set_bg(BG);
            }
            for y in map_area.y..map_area.bottom() {
                buf[(map_area.x, y)].set_symbol("│");
                buf[(map_area.x, y)].set_fg(BTXT);
                buf[(map_area.x, y)].set_bg(BG);
                buf[(map_area.right() - 1, y)].set_symbol("│");
                buf[(map_area.right() - 1, y)].set_fg(BTXT);
                buf[(map_area.right() - 1, y)].set_bg(BG);
            }

            // Corners
            let corners = [
                (map_area.x, map_area.y, "┌"),
                (map_area.right() - 1, map_area.y, "┐"),
                (map_area.x, map_area.bottom() - 1, "└"),
                (map_area.right() - 1, map_area.bottom() - 1, "┘"),
            ];
            for &(cx, cy, sym) in &corners {
                buf[(cx, cy)].set_symbol(sym);
                buf[(cx, cy)].set_fg(BTXT);
                buf[(cx, cy)].set_bg(BG);
            }

            // Cardinal markers
            let cards = [
                (map_area.x + 2, map_area.y, "000°"),
                (map_area.right().saturating_sub(5), map_area.y, "090°"),
                (map_area.x + 2, map_area.bottom() - 1, "180°"),
                (map_area.right().saturating_sub(5), map_area.bottom() - 1, "270°"),
            ];
            for &(cx, cy, label) in &cards {
                for (i, ch) in label.chars().enumerate() {
                    let cell = &mut buf[(cx + i as u16, cy)];
                    cell.set_symbol(&ch.to_string());
                    cell.set_fg(DTXT);
                    cell.set_bg(BG);
                }
            }

            // Face diamond
            if dot_x >= map_area.x + 1
                && dot_x < map_area.right() - 1
                && dot_y >= map_area.y + 1
                && dot_y < map_area.bottom() - 1
            {
                let uc = self.s.kalman_uncertainty;
                let dc = if uc < 0.3 { TXT } else if uc < 0.6 { YLW } else { RED };
                buf[(dot_x, dot_y)].set_symbol("◇");
                buf[(dot_x, dot_y)].set_fg(dc);
                buf[(dot_x, dot_y)].set_bg(BG);
                buf[(dot_x, dot_y)]
                    .set_style(Style::default().add_modifier(Modifier::BOLD));
            }

            // Info line
            let fr = face_ratio(&self.s);
            let off_x = self.s.face_x - fw / 2.0;
            let off_y = self.s.face_y - fh / 2.0;
            let off_pct_x = (off_x / fw * 200.0) as i64;
            let off_pct_y = (off_y / fh * 200.0) as i64;
            let dir_ind: String = if off_x.abs() < 5.0 && off_y.abs() < 5.0 {
                "[CENTER]".into()
            } else {
                format!("[{:>+3}% {:>+3}%]", off_pct_x, off_pct_y)
            };
            let bb = format!(
                "  [FACE] {}  CNF:{:.3}  KAL:{:.3}  SIZE:{:.1}%",
                dir_ind, self.s.face_confidence, self.s.kalman_uncertainty, fr
            );
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(bb, Style::default().fg(TXT))))
                    .style(Style::default().bg(BG)),
                info_area,
            );
        } else {
            f.render_widget(
                Paragraph::new("  [X] NO FACE DETECTED")
                    .alignment(Alignment::Center)
                    .style(
                        Style::default()
                            .fg(RED)
                            .add_modifier(Modifier::BOLD),
                    ),
                map_area,
            );
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!(
                        "  KALMAN: {}  CONF:{:.3}",
                        gauge(
                            (self.s.kalman_uncertainty * 100.0).min(100.0) as u16,
                            10
                        ),
                        self.s.face_confidence
                    ),
                    Style::default().fg(DTXT),
                )))
                .style(Style::default().bg(BG)),
                info_area,
            );
        }
    }

    // ── SYSTEM STATE ──
    fn render_system_state(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }
        for (i, ch) in " SYSTEM STATE ".chars().enumerate() {
            let cx = a.x + 2 + i as u16;
            if cx < a.right() {
                buf[(cx, a.y)].set_symbol(&ch.to_string());
                buf[(cx, a.y)].set_fg(TXT);
                buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
            }
        }

        let inner = Rect::new(a.x, a.y + 1, a.width, a.height - 1);
        let dw = inner.width as usize;

        let lock_st = if self.s.mode == "LOCKED" {
            "ARM"
        } else {
            "OFF"
        };
        let rec_st = status_bool(self.s.recording, "ON", "OFF");
        let target_st = if self.s.mode == "TRACKING_HAND" {
            "HAND"
        } else if self.s.face_detected {
            "FACE"
        } else {
            "NONE"
        };
        let gesture_display = if self.s.gesture != "NONE" {
            self.s.gesture.as_str()
        } else {
            "NONE"
        };

        let items: Vec<(&str, String, Color)> = vec![
            ("MODE", self.s.mode.clone(), mode_color(&self.s.mode)),
            (
                "SUBJECT",
                target_st.into(),
                if target_st == "NONE" {
                    DTXT
                } else {
                    TXT
                },
            ),
            ("LOCK", lock_st.into(), if lock_st == "ARM" { YLW } else { DTXT }),
            ("REC", rec_st.into(), if self.s.recording { RED } else { DTXT }),
            (
                "ZOOM",
                format!("{:.1}X", self.s.zoom_level),
                YLW,
            ),
            (
                "GESTURE",
                gesture_display.into(),
                if self.s.gesture != "NONE" {
                    YLW
                } else {
                    DTXT
                },
            ),
            (
                "FACE CONF",
                format!("{:.3}", self.s.face_confidence),
                if self.s.face_confidence > 0.7 {
                    TXT
                } else if self.s.face_confidence > 0.4 {
                    YLW
                } else {
                    RED
                },
            ),
            (
                "KALMAN UNC",
                format!("{:.3}", self.s.kalman_uncertainty),
                if self.s.kalman_uncertainty < 0.3 {
                    TXT
                } else if self.s.kalman_uncertainty < 0.6 {
                    YLW
                } else {
                    RED
                },
            ),
            (
                "FACE SIZE",
                format!("{:.1}%", face_ratio(&self.s)),
                TXT,
            ),
        ];

        let mut lines: Vec<Line> = items
            .iter()
            .map(|(label, value, color)| {
                let vw = value.len();
                let dots = dw.saturating_sub(label.len() + vw + 4);
                let d = ".".repeat(dots);
                Line::from(vec![
                    Span::styled(format!("  {}", label), Style::default().fg(DTXT)),
                    Span::styled(d, Style::default().fg(DTXT)),
                    Span::styled(
                        format!(" {}", value),
                        Style::default()
                            .fg(*color)
                            .add_modifier(Modifier::BOLD),
                    ),
                ])
            })
            .collect();

        // IMU data below if there's room
        if lines.len() as u16 + 3 < inner.height {
            lines.push(Line::from(Span::styled(
                "  ── IMU ORIENTATION ──",
                Style::default().fg(DTXT),
            )));
            let imu_dot =
                dw.saturating_sub("IMU PITCH".len() + 18).max(1);
            lines.push(Line::from(vec![
                Span::styled("  IMU PITCH", Style::default().fg(DTXT)),
                Span::styled(
                    format!(
                        "{} {:>+.1}°",
                        ".".repeat(imu_dot),
                        self.s.imu_pitch
                    ),
                    Style::default().fg(TXT),
                ),
            ]));
            if lines.len() as u16 + 1 < inner.height {
                lines.push(Line::from(vec![
                    Span::styled("  IMU ROLL", Style::default().fg(DTXT)),
                    Span::styled(
                        format!(
                            "{} {:>+.1}°",
                            ".".repeat(imu_dot),
                            self.s.imu_roll
                        ),
                        Style::default().fg(TXT),
                    ),
                ]));
            }
            if lines.len() as u16 + 1 < inner.height {
                lines.push(Line::from(vec![
                    Span::styled("  IMU YAW", Style::default().fg(DTXT)),
                    Span::styled(
                        format!(
                            "{} {:>+.1}°",
                            ".".repeat(imu_dot),
                            self.s.imu_yaw
                        ),
                        Style::default().fg(DTXT),
                    ),
                ]));
            }
        }

        f.render_widget(Paragraph::new(lines).style(Style::default().bg(BG)), inner);
    }

    // ── AXIS CONTROL ──
    fn render_axis_control(&self, f: &mut ratatui::Frame, a: Rect) {
        // Draw title line to buffer
        {
            let buf = f.buffer_mut();
            for x in a.x..a.right() {
                buf[(x, a.y)].set_symbol("─");
                buf[(x, a.y)].set_fg(BTXT);
            }
            for (i, ch) in " AXIS CONTROL ".chars().enumerate() {
                let cx = a.x + 2 + i as u16;
                if cx < a.right() {
                    buf[(cx, a.y)].set_symbol(&ch.to_string());
                    buf[(cx, a.y)].set_fg(TXT);
                    buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
                }
            }
        }

        let inner = Rect::new(a.x, a.y + 1, a.width, a.height - 1);
        if inner.height < 4 { return; }

        let h2 = inner.height / 2;
        let areas = [
            Rect::new(inner.x, inner.y, inner.width, h2),
            Rect::new(inner.x, inner.y + h2, inner.width, inner.height - h2),
        ];

        let axes: [(&str, f64, String, String); 2] = [
            ("PAN", self.s.pid_pan_error,
             format!("P:{} I:{} D:{}", fmt_pid(self.s.pid_pan_p), fmt_pid(self.s.pid_pan_i), fmt_pid(self.s.pid_pan_d)),
             format!("ERR:{}  OUT:{:+.1}°  TGT:{:>3}°", fmt_pid(self.s.pid_pan_error), self.s.pid_pan_output, self.s.pan_target)),
            ("TLT", self.s.pid_tilt_error,
             format!("P:{} I:{} D:{}", fmt_pid(self.s.pid_tilt_p), fmt_pid(self.s.pid_tilt_i), fmt_pid(self.s.pid_tilt_d)),
             format!("ERR:{}  OUT:{:+.1}°  TGT:{:>3}°", fmt_pid(self.s.pid_tilt_error), self.s.pid_tilt_output, self.s.tilt_target)),
        ];

        let bar_w = (inner.width as usize).saturating_sub(11).max(3);

        // Phase 1: All buffer draws (labels + bars + numbers)
        {
            let buf = f.buffer_mut();
            for (idx, (label, err_val, _, _)) in axes.iter().enumerate() {
                let ar = areas[idx];
                // Label on first row
                let lbl = format!(" {} ", label);
                for (i, ch) in lbl.chars().enumerate() {
                    let cx = ar.x + i as u16;
                    if cx < ar.right() {
                        buf[(cx, ar.y)].set_symbol(&ch.to_string());
                        buf[(cx, ar.y)].set_fg(TXT);
                        buf[(cx, ar.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
                    }
                }
                // Bar row: error magnitude (capped at 1.0 for visual reference)
                let mag = err_val.abs().min(1.0);
                let pct = (mag * 100.0) as u16;
                let filled = (pct as usize * bar_w / 100).min(bar_w);
                let empty = bar_w.saturating_sub(filled);
                let bar_s: String = format!("{}{}", "▓".repeat(filled), "░".repeat(empty));
                let bar_line = format!("  {:.3} {}", err_val, bar_s);
                let cy = ar.y + 1;
                for (ci, ch) in bar_line.chars().enumerate() {
                    let cx = ar.x + ci as u16;
                    if cx < ar.right() && cy < ar.bottom() {
                        buf[(cx, cy)].set_symbol(&ch.to_string());
                        buf[(cx, cy)].set_bg(BG);
                        if ci < 5 {
                            buf[(cx, cy)].set_fg(YLW);
                        } else if ci < 6 + filled {
                            buf[(cx, cy)].set_fg(TXT);
                        } else {
                            buf[(cx, cy)].set_fg(DTXT);
                        }
                    }
                }
                // Separator dot row
                if ar.y + 2 < ar.bottom() {
                    let sy = ar.y + 2;
                    for ci in 0..inner.width as usize {
                        let cx = ar.x + ci as u16;
                        if cx < ar.right() {
                            buf[(cx, sy)].set_symbol("·");
                            buf[(cx, sy)].set_fg(DTXT);
                            buf[(cx, sy)].set_bg(BG);
                        }
                    }
                }
            }
        }

        // Phase 2: All widget renders
        for (idx, (_, _, line1, line2)) in axes.iter().enumerate() {
            let ar = areas[idx];
            let p1 = Paragraph::new(Line::from(Span::styled(line1, Style::default().fg(DTXT))))
                .style(Style::default().bg(BG));
            let l1y = ar.y + 3;
            if l1y < ar.bottom() {
                f.render_widget(p1, Rect::new(ar.x + 1, l1y, ar.width.saturating_sub(2), 1));
            }
            if ar.y + 4 < ar.bottom() {
                let p2 = Paragraph::new(Line::from(Span::styled(line2, Style::default().fg(TXT))))
                    .style(Style::default().bg(BG));
                f.render_widget(p2, Rect::new(ar.x + 1, ar.y + 4, ar.width.saturating_sub(2), 1));
            }
        }
    }

    // ── SIGNAL ANALYSIS ──
    fn render_signal_analysis(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }
        for (i, ch) in " SIGNAL ANALYSIS ".chars().enumerate() {
            let cx = a.x + 2 + i as u16;
            if cx < a.right() {
                buf[(cx, a.y)].set_symbol(&ch.to_string());
                buf[(cx, a.y)].set_fg(TXT);
                buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
            }
        }

        let inner = Rect::new(a.x, a.y + 1, a.width, a.height - 1);
        if inner.height < 2 { return; }

        let lat = &self.s.latency_ms;
        let stages = ["capture", "detect", "gesture", "track", "pid", "display", "ipc"];
        let labels = ["CAP", "DET", "GES", "TRK", "PID", "DISP", "IPC"];

        let mut vals: Vec<f64> = Vec::new();
        let mut total = 0.0_f64;
        for k in &stages {
            let v = lat.as_ref().and_then(|m| m.get(*k)).copied().unwrap_or(0.0);
            vals.push(v);
            total += v;
        }
        let max_v = vals.iter().cloned().fold(0.0_f64, f64::max).max(1.0);

        let bar_w = (inner.width as usize).saturating_sub(14);
        let rows_avail = inner.height as usize;
        let mut row = 0;

        // Bar chart rows with visual separation
        for (i, (lb, &v)) in labels.iter().zip(vals.iter()).enumerate() {
            if row >= rows_avail { break; }
            let filled = (bar_w as f64 * v / max_v).ceil() as usize;
            let empty = bar_w.saturating_sub(filled);
            let line_s = format!(" {} {}{} {:4.0}MS",
                lb,
                "▓".repeat(filled),
                "░".repeat(empty),
                v);
            let cy = inner.y + row as u16;
            for (ci, ch) in line_s.chars().enumerate() {
                let cx = inner.x + ci as u16;
                if cx < a.right() {
                    buf[(cx, cy)].set_symbol(&ch.to_string());
                    buf[(cx, cy)].set_bg(BG);
                    if ci <= 4 {
                        buf[(cx, cy)].set_fg(DTXT);
                    } else if ci > bar_w + 5 {
                        buf[(cx, cy)].set_fg(YLW);
                    } else if filled > 0 && ci <= 5 + filled {
                        buf[(cx, cy)].set_fg(TXT);
                    } else {
                        buf[(cx, cy)].set_fg(DTXT);
                    }
                }
            }
            row += 1;
            // Dotted separator between bars
            if row < rows_avail && i + 1 < labels.len() {
                let cy = inner.y + row as u16;
                for ci in 0..inner.width as usize {
                    let cx = inner.x + ci as u16;
                    if cx < a.right() {
                        buf[(cx, cy)].set_symbol("·");
                        buf[(cx, cy)].set_fg(DTXT);
                        buf[(cx, cy)].set_bg(BG);
                    }
                }
                row += 1;
            }
        }

        // Separator + total / fps
        if row < rows_avail {
            let tc = if total < 20.0 { TXT } else if total < 40.0 { YLW } else { RED };
            let sep_s = format!("{}", "─".repeat(inner.width as usize));
            let cy = inner.y + row as u16;
            for (ci, ch) in sep_s.chars().enumerate() {
                let cx = inner.x + ci as u16;
                if cx < a.right() {
                    buf[(cx, cy)].set_symbol(&ch.to_string());
                    buf[(cx, cy)].set_fg(BTXT);
                    buf[(cx, cy)].set_bg(BG);
                }
            }
            row += 1;

            if row < rows_avail {
                let sum_l = format!(" LAT:{}MS  FPS:{:.1}  F#{}", total as usize, self.s.fps, self.frames);
                let tl = Paragraph::new(Line::from(Span::styled(&sum_l, Style::default().fg(tc))))
                    .style(Style::default().bg(BG));
                f.render_widget(tl, Rect::new(inner.x, inner.y + row as u16, inner.width, 1));
                row += 1;
            }

            if row < rows_avail {
                let fps_pct = (self.s.fps * 100.0 / 30.0).min(100.0) as u16;
                let fps_c = if self.s.fps > 20.0 { TXT } else if self.s.fps > 10.0 { YLW } else { RED };
                let fps_s = format!(" FPS: {}", gauge(fps_pct, (inner.width as usize).saturating_sub(6)));
                let fl = Paragraph::new(Line::from(Span::styled(&fps_s, Style::default().fg(fps_c).add_modifier(Modifier::BOLD))))
                    .style(Style::default().bg(BG));
                f.render_widget(fl, Rect::new(inner.x, inner.y + row as u16, inner.width, 1));
            }
        }
    }

    // ── EVENT LOG ──
    fn render_event_log(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }
        for (i, ch) in " EVENT LOG ".chars().enumerate() {
            let cx = a.x + 2 + i as u16;
            if cx < a.right() {
                buf[(cx, a.y)].set_symbol(&ch.to_string());
                buf[(cx, a.y)].set_fg(TXT);
                buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
            }
        }

        let inner = Rect::new(a.x, a.y + 1, a.width, a.height - 1);
        let mut rows: Vec<Line> = Vec::new();

        // Table header
        let max_event_w = (inner.width as usize).saturating_sub(12);
        rows.push(Line::from(vec![
            Span::styled("  TIME     EVENT", Style::default().fg(DTXT)),
        ]));

        for e in self.s.events.iter().rev().take(EVT_SHOW) {
            let (ts, rest) = if e.len() > 9 { e.split_at(9) } else { ("", e.as_str()) };
            let disp = if rest.len() > max_event_w {
                &rest[..max_event_w.saturating_sub(3)]
            } else {
                rest
            };
            rows.push(Line::from(vec![
                Span::styled(format!("  {}", ts), Style::default().fg(DTXT)),
                Span::styled(disp.to_string(), Style::default().fg(TXT)),
            ]));
        }
        if rows.len() == 1 {
            rows.push(Line::from(Span::styled(
                "  -- NO EVENTS --",
                Style::default().fg(DTXT),
            )));
        }

        f.render_widget(Paragraph::new(rows).style(Style::default().bg(BG)), inner);
    }

    // ── SIGNAL LEVELS ──
    fn render_signal_levels(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }
        for (i, ch) in " SIGNAL LEVELS ".chars().enumerate() {
            let cx = a.x + 2 + i as u16;
            if cx < a.right() {
                buf[(cx, a.y)].set_symbol(&ch.to_string());
                buf[(cx, a.y)].set_fg(TXT);
                buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
            }
        }

        let inner_y = a.y + 1;
        let inner_h = a.height.saturating_sub(1);
        if inner_h < 3 { return; }

        let tanks: [(&str, u16, Color); 4] = [
            ("FCE", (self.s.face_confidence * 100.0) as u16,
             if self.s.face_confidence > 0.7 { TXT } else if self.s.face_confidence > 0.4 { YLW } else { RED }),
            ("KAL", ((1.0 - self.s.kalman_uncertainty.min(1.0)) * 100.0) as u16,
             if self.s.kalman_uncertainty < 0.3 { TXT } else if self.s.kalman_uncertainty < 0.6 { YLW } else { RED }),
            ("FPS", (self.s.fps * 100.0 / 30.0).min(100.0) as u16,
             if self.s.fps > 20.0 { TXT } else if self.s.fps > 10.0 { YLW } else { RED }),
            ("PID", ((1.0 - (self.s.pid_pan_error.abs() + self.s.pid_tilt_error.abs()).min(1.0)) * 100.0) as u16,
             if self.s.pid_pan_error.abs() + self.s.pid_tilt_error.abs() < 0.3 { TXT } else if self.s.pid_pan_error.abs() + self.s.pid_tilt_error.abs() < 0.6 { YLW } else { RED }),
        ];

        let n = tanks.len();
        let gap = 1u16;
        let tank_w = 5u16;
        let total_w = (tank_w + gap) * n as u16 - gap;
        let x_off = a.x + (a.width.saturating_sub(total_w)) / 2;
        let gauge_h = inner_h.saturating_sub(2).min(5);

        for (ti, (name, val, col)) in tanks.iter().enumerate() {
            let tx = x_off + ti as u16 * (tank_w + gap);

            // Label centered
            let lx = tx + (tank_w.saturating_sub(name.len() as u16)) / 2;
            for (i, ch) in name.chars().enumerate() {
                let c = &mut buf[(lx + i as u16, inner_y)];
                c.set_symbol(&ch.to_string());
                c.set_fg(*col);
                c.set_bg(BG);
                c.set_style(Style::default().add_modifier(Modifier::BOLD));
            }

            // Value below
            let vs = format!("{}%", val);
            let vx = tx + (tank_w.saturating_sub(vs.len() as u16)) / 2;
            for (i, ch) in vs.chars().enumerate() {
                let c = &mut buf[(vx + i as u16, inner_y + 1)];
                c.set_symbol(&ch.to_string());
                c.set_fg(YLW);
                c.set_bg(BG);
            }

            // Vertical bar (bottom-up)
            let fill = (*val as usize * gauge_h as usize / 100).min(gauge_h as usize);
            for row in 0..gauge_h {
                let cy = inner_y + 2 + (gauge_h - 1 - row);
                let filled = (row as usize) < fill;
                let sym = if filled { "█" } else { "░" };
                for cx in tx..(tx + tank_w).min(a.right()) {
                    buf[(cx, cy)].set_symbol(sym);
                    buf[(cx, cy)].set_fg(if filled { *col } else { DTXT });
                    buf[(cx, cy)].set_bg(BG);
                }
            }

            // Separator between tanks
            if ti + 1 < n {
                let sx = tx + tank_w;
                for sy in inner_y..(inner_y + 2 + gauge_h).min(a.bottom()) {
                    let c = &mut buf[(sx, sy)];
                    if sy == inner_y || sy == inner_y + 1 {
                        c.set_symbol("│");
                    } else {
                        c.set_symbol("┊");
                    }
                    c.set_fg(DTXT);
                    c.set_bg(BG);
                }
            }
        }
    }

    // ── ACTUATOR STATE ──
    fn render_actuator_state(&self, f: &mut ratatui::Frame, a: Rect) {
        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }
        for (i, ch) in " ACTUATOR STATE ".chars().enumerate() {
            let cx = a.x + 2 + i as u16;
            if cx < a.right() {
                buf[(cx, a.y)].set_symbol(&ch.to_string());
                buf[(cx, a.y)].set_fg(TXT);
                buf[(cx, a.y)].set_style(Style::default().add_modifier(Modifier::BOLD));
            }
        }

        let inner = Rect::new(a.x, a.y + 1, a.width, a.height - 1);
        if inner.height < 2 { return; }

        // Build bar lines using the same style as SIGNAL ANALYSIS
        let mut lines: Vec<Line> = Vec::new();
        let w = inner.width as usize;
        let bar_w = (w / 2).saturating_sub(10).max(3);

        // Row 1: Position
        {
            let p_a = self.s.pan_angle;
            let t_a = self.s.tilt_angle;
            let p_t = self.s.pan_target;
            let t_t = self.s.tilt_target;
            let pd = (p_t - p_a).abs();
            let td = (t_t - t_a).abs();
            let pc = if pd < 5 { TXT } else if pd < 15 { YLW } else { RED };
            let tc = if td < 5 { TXT } else if td < 15 { YLW } else { RED };
            let p_fill = (pd.min(180) as usize * bar_w / 180).min(bar_w);
            let p_bar = format!("{}{}", "▓".repeat(p_fill), "░".repeat(bar_w.saturating_sub(p_fill)));
            let t_fill = (td.min(180) as usize * bar_w / 180).min(bar_w);
            let t_bar = format!("{}{}", "▓".repeat(t_fill), "░".repeat(bar_w.saturating_sub(t_fill)));
            lines.push(Line::from(vec![
                Span::styled("  PAN", Style::default().fg(DTXT)),
                Span::styled(p_bar.clone(), Style::default().fg(pc)),
                Span::styled(format!(" {:>3}°→{:<3}", p_a, p_t), Style::default().fg(pc).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled("TLT", Style::default().fg(DTXT)),
                Span::styled(t_bar, Style::default().fg(tc)),
                Span::styled(format!(" {:>3}°→{:<3}", t_a, t_t), Style::default().fg(tc).add_modifier(Modifier::BOLD)),
            ]));
        }

        if inner.height >= 3 {
            lines.push(Line::from(Span::styled(
                "·".repeat(w.saturating_sub(2)),
                Style::default().fg(DTXT),
            )));
        }

        // Row 2: Error
        if inner.height >= 4 {
            let pe = self.s.pid_pan_error.abs().min(1.0);
            let te = self.s.pid_tilt_error.abs().min(1.0);
            let pc = if pe < 0.1 { TXT } else if pe < 0.3 { YLW } else { RED };
            let tc = if te < 0.1 { TXT } else if te < 0.3 { YLW } else { RED };
            let p_fill = (pe * bar_w as f64) as usize;
            let p_bar = format!("{}{}", "▓".repeat(p_fill), "░".repeat(bar_w.saturating_sub(p_fill)));
            let t_fill = (te * bar_w as f64) as usize;
            let t_bar = format!("{}{}", "▓".repeat(t_fill), "░".repeat(bar_w.saturating_sub(t_fill)));
            lines.push(Line::from(vec![
                Span::styled("  ERR", Style::default().fg(DTXT)),
                Span::styled(p_bar, Style::default().fg(pc)),
                Span::styled(format!(" {:+.3}", self.s.pid_pan_error), Style::default().fg(YLW)),
                Span::raw("  "),
                Span::styled("TLT", Style::default().fg(DTXT)),
                Span::styled(t_bar, Style::default().fg(tc)),
                Span::styled(format!(" {:+.3}", self.s.pid_tilt_error), Style::default().fg(YLW)),
            ]));
        }

        if inner.height >= 5 {
            lines.push(Line::from(Span::styled(
                "·".repeat(w.saturating_sub(2)),
                Style::default().fg(DTXT),
            )));
        }

        // Row 3: Output
        if inner.height >= 6 {
            let po = (self.s.pid_pan_output.abs() / 30.0).min(1.0);
            let to = (self.s.pid_tilt_output.abs() / 30.0).min(1.0);
            let pc = if po < 0.3 { TXT } else if po < 0.6 { YLW } else { RED };
            let tc = if to < 0.3 { TXT } else if to < 0.6 { YLW } else { RED };
            let p_fill = (po * bar_w as f64) as usize;
            let p_bar = format!("{}{}", "▓".repeat(p_fill), "░".repeat(bar_w.saturating_sub(p_fill)));
            let t_fill = (to * bar_w as f64) as usize;
            let t_bar = format!("{}{}", "▓".repeat(t_fill), "░".repeat(bar_w.saturating_sub(t_fill)));
            lines.push(Line::from(vec![
                Span::styled("  OUT", Style::default().fg(DTXT)),
                Span::styled(p_bar, Style::default().fg(pc)),
                Span::styled(format!(" {:+.1}°", self.s.pid_pan_output), Style::default().fg(YLW)),
                Span::raw("  "),
                Span::styled("TLT", Style::default().fg(DTXT)),
                Span::styled(t_bar, Style::default().fg(tc)),
                Span::styled(format!(" {:+.1}°", self.s.pid_tilt_output), Style::default().fg(YLW)),
            ]));
        }

        if inner.height >= 7 {
            lines.push(Line::from(Span::styled(
                "·".repeat(w.saturating_sub(2)),
                Style::default().fg(DTXT),
            )));
        }

        // Row 4: IMU + toggles
        if inner.height >= 8 {
            let mode_d = self.s.mode.as_str();
            let lock_d = if mode_d == "LOCKED" { "ARM" } else { "OFF" };
            let rec_d = if self.s.recording { "ON" } else { "OFF" };
            let subj = if mode_d == "TRACKING_HAND" { "HAND" } else if self.s.face_detected { "FACE" } else { "NONE" };
            let ges_m = if self.s.gesture_method.is_empty() { "N/A" } else { &self.s.gesture_method };
            lines.push(Line::from(vec![
                Span::styled("  IMU", Style::default().fg(DTXT)),
                Span::styled(format!(" P:{:>+.1}", self.s.imu_pitch), Style::default().fg(TXT)),
                Span::styled(format!(" R:{:>+.1}", self.s.imu_roll), Style::default().fg(TXT)),
                Span::styled(format!(" Y:{:>+.1}", self.s.imu_yaw), Style::default().fg(DTXT)),
                Span::raw(" "),
                Span::styled("E:", Style::default().fg(DTXT)),
                Span::styled(format!("P{}%", edge_dist(self.s.pan_angle, 0, 180)), Style::default().fg(
                    if edge_dist(self.s.pan_angle, 0, 180) < 15 { YLW } else { TXT })),
                Span::styled(format!(" T{}%", edge_dist(self.s.tilt_angle, 0, 180)), Style::default().fg(
                    if edge_dist(self.s.tilt_angle, 0, 180) < 15 { YLW } else { TXT })),
                Span::raw("  "),
                Span::styled(format!("[{}]", mode_d), Style::default().fg(mode_color(mode_d)).add_modifier(Modifier::BOLD)),
                Span::styled(format!("[{}]", lock_d), Style::default().fg(YLW)),
                Span::styled(format!("[{}]", rec_d), Style::default().fg(if self.s.recording { RED } else { DTXT })),
                Span::styled(format!("[{:.1}X]", self.s.zoom_level), Style::default().fg(YLW)),
                Span::styled(format!("[{}]", self.s.gesture), Style::default().fg(if self.s.gesture != "NONE" { YLW } else { DTXT })),
                Span::styled(format!("[{}]", subj), Style::default().fg(TXT)),
                Span::raw(" "),
                Span::styled(format!("GES:{} F#{:.0}", ges_m.to_uppercase(), self.frames as f64), Style::default().fg(DTXT)),
            ]));
        }

        f.render_widget(Paragraph::new(lines).style(Style::default().bg(BG)), inner);
    }

    // ── FOOTER ──
    fn render_footer(&self, f: &mut ratatui::Frame, a: Rect) {
        let [cl, sl] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(a);

        let buf = f.buffer_mut();
        for x in a.x..a.right() {
            buf[(x, a.y)].set_symbol("─");
            buf[(x, a.y)].set_fg(BTXT);
        }

        // Control bindings
        let ctrl = Line::from(vec![
            Span::styled(" Q:QUIT", Style::default().fg(TXT)),
            Span::styled("  H:HOME", Style::default().fg(TXT)),
            Span::styled("  SPC:LOCK", Style::default().fg(TXT)),
            Span::styled("  R:REC", Style::default().fg(TXT)),
            Span::styled("  │  ", Style::default().fg(BTXT)),
            Span::styled(
                format!("  FPS:{:.1}  AVG:{:.1}", self.s.fps, self.bufs.avg_fps()),
                Style::default().fg(TXT),
            ),
            Span::styled("  │  ", Style::default().fg(BTXT)),
            Span::styled(
                format!(
                    "  GIMBAL P:{:>3}° T:{:>3}°",
                    self.s.pan_angle, self.s.tilt_angle
                ),
                Style::default().fg(DTXT),
            ),
            Span::styled("  │  ", Style::default().fg(BTXT)),
            Span::styled(
                format!("  EDGE P:{}% T:{}%", edge_dist(self.s.pan_angle, 0, 180), edge_dist(self.s.tilt_angle, 0, 180)),
                Style::default().fg(DTXT),
            ),
        ]);
        f.render_widget(
            Paragraph::new(ctrl).style(Style::default().bg(BG)),
            cl,
        );

        // Latency breakdown
        let lat = &self.s.latency_ms;
        let total: f64 = lat.as_ref().map(|m| m.values().sum()).unwrap_or(0.0);
        let tc = if total < 20.0 {
            TXT
        } else if total < 40.0 {
            YLW
        } else {
            RED
        };
        let mut lat_str = format!("LAT:{:5.1}MS", total);
        let order = ["detect", "capture", "gesture", "display", "ipc", "track", "pid"];
        let labels = ["DET", "CAP", "GES", "DIS", "IPC", "TRK", "PID"];
        for (lb, k) in labels.iter().zip(order.iter()) {
            if let Some(v) = lat.as_ref().and_then(|m| m.get(*k)) {
                lat_str.push_str(&format!("  {}:{:>4.0}", lb, v));
            }
        }
        let latency_line = Line::from(vec![
            Span::styled(lat_str, Style::default().fg(tc)),
            Span::styled("  │  ", Style::default().fg(BTXT)),
            Span::styled(
                format!("ZOOM:{:.1}X", self.s.zoom_level),
                Style::default().fg(YLW),
            ),
            Span::styled("  │  ", Style::default().fg(BTXT)),
            Span::styled(
                format!("MODE:{}  F#{:<6}", self.s.mode, self.frames),
                Style::default().fg(mode_color(&self.s.mode)),
            ),
        ]);
        f.render_widget(
            Paragraph::new(latency_line).style(Style::default().bg(BG)),
            sl,
        );
    }
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut so = io::stdout();
    execute!(so, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(so);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    let mut app = App::new();
    let result = app.run(&mut terminal);
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    if let Err(e) = &result {
        eprintln!("Error: {}", e);
    }
    result
}
