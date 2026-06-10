use std::cell::RefCell;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use gtk4::gdk::Display;
use gtk4::gdk_pixbuf::Pixbuf;
use gtk4::{
    Align, Box, Button, CssProvider, DrawingArea, EventControllerKey, Label, Overlay, Picture,
    ToggleButton,
};
use libadwaita::prelude::*;
use libadwaita::{Application, ApplicationWindow, HeaderBar};

use neuracam_shared::{read_msg, NeuraCamState, MSG_FRAME, MSG_STATE};

const SOCKET_PATH: &str = "/tmp/neuracam.sock";
const INPUT_SOCK_PATH: &str = "/tmp/neuracam_input.sock";
const RECONNECT_DELAY: Duration = Duration::from_secs(2);
const CSS: &str = "
    .osd-label { font-size: 14px; font-weight: bold; color: #cccccc; background: rgba(0,0,0,0.5); padding: 3px 8px; border-radius: 4px; }
    .osd-mode { font-size: 17px; }
    window { background: #0d0d0d; }
    .feed-area { background: #0d0d0d; }
    .toolbar-btn { min-width: 32px; min-height: 32px; border-radius: 4px; margin: 2px; }
    .toolbar-btn:hover { background: rgba(255,255,255,0.08); }
    .toolbar-btn:checked { background: rgba(255,255,255,0.15); }
    .info-label { font-size: 12px; color: #666666; padding: 2px 6px; }
";

fn mode_color(m: &str) -> &str {
    match m {
        "TRACKING" => "#cccccc",
        "TRACKING_HAND" => "#d4a040",
        "LOCKED" => "#d4a040",
        "IDLE" => "#666666",
        "SEARCH" => "#d4a040",
        "HOME" => "#80aacc",
        _ => "#cccccc",
    }
}

fn send_key(key: char) {
    thread::spawn(move || {
        if let Ok(mut s) = UnixStream::connect(INPUT_SOCK_PATH) {
            let _ = s.write_all(&[key as u8]);
        }
    });
}

fn load_css() {
    let p = CssProvider::new();
    p.load_from_data(CSS);
    if let Some(d) = Display::default() {
        gtk4::style_context_add_provider_for_display(
            &d,
            &p,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

fn draw_face_overlay(da: &DrawingArea, state: &NeuraCamState, pw: f64, ph: f64) {
    if !state.face_detected {
        da.set_draw_func(|_, _, _, _| {});
        return;
    }
    let state = state.clone();
    da.set_draw_func(move |_, cr, w, h| {
        let sx = w as f64 / pw;
        let sy = h as f64 / ph;
        let fx = (state.face_x - state.face_w / 2.0) * sx;
        let fy = (state.face_y - state.face_h / 2.0) * sy;
        let fw = state.face_w * sx;
        let fh = state.face_h * sy;

        cr.set_source_rgba(0.6, 0.6, 0.6, 0.8);
        cr.set_line_width(1.5);
        cr.rectangle(fx, fy, fw, fh);
        let _ = cr.stroke();

        let cx = w as f64 / 2.0;
        let cy = h as f64 / 2.0;
        cr.set_source_rgba(0.4, 0.4, 0.4, 0.5);
        cr.set_line_width(1.0);
        cr.move_to(cx - 15.0, cy);
        cr.line_to(cx + 15.0, cy);
        cr.move_to(cx, cy - 15.0);
        cr.line_to(cx, cy + 15.0);
        let _ = cr.stroke();

        let uc = state.kalman_uncertainty;
        let dc = if uc < 0.3 {
            (0.6, 0.6, 0.6)
        } else if uc < 0.6 {
            (0.8, 0.6, 0.25)
        } else {
            (0.8, 0.3, 0.3)
        };
        let dx = state.face_x * sx;
        let dy = state.face_y * sy;
        cr.set_source_rgba(dc.0, dc.1, dc.2, 0.9);
        cr.set_line_width(2.0);
        cr.move_to(dx, dy - 4.0);
        cr.line_to(dx + 4.0, dy);
        cr.line_to(dx, dy + 4.0);
        cr.line_to(dx - 4.0, dy);
        cr.close_path();
        let _ = cr.stroke();
    });
}

fn main() {
    libadwaita::init().expect("Failed to init Adwaita");
    load_css();

    let app = Application::builder()
        .application_id("com.neuracam.viewer")
        .build();

    app.connect_activate(|app| {
        let show_osd = Arc::new(AtomicBool::new(true));

        // ── OSD ──
        let osd_mode = Label::builder()
            .css_classes(["osd-label", "osd-mode"])
            .label("[ IDLE ]")
            .halign(Align::Start)
            .valign(Align::Start)
            .margin_start(10)
            .margin_top(10)
            .build();
        let osd_fps = Label::builder()
            .css_classes(["osd-label"])
            .label("")
            .halign(Align::Start)
            .valign(Align::Start)
            .margin_start(10)
            .margin_top(40)
            .build();
        let osd_info = Label::builder()
            .css_classes(["osd-label"])
            .label("")
            .halign(Align::Start)
            .valign(Align::Start)
            .margin_start(10)
            .margin_top(64)
            .build();

        let osd_box = Box::builder().build();
        osd_box.append(&osd_mode);
        osd_box.append(&osd_fps);
        osd_box.append(&osd_info);

        // ── Camera feed ──
        let picture = Picture::builder()
            .keep_aspect_ratio(true)
            .can_shrink(true)
            .halign(Align::Center)
            .valign(Align::Center)
            .hexpand(true)
            .vexpand(true)
            .css_classes(["feed-area"])
            .build();
        let face_da = DrawingArea::builder()
            .halign(Align::Fill)
            .valign(Align::Fill)
            .hexpand(true)
            .vexpand(true)
            .build();

        let overlay = Overlay::new();
        overlay.set_child(Some(&picture));
        overlay.add_overlay(&face_da);
        overlay.add_overlay(&osd_box);

        // ── Toolbar ──
        let btn_home = Button::builder()
            .icon_name("go-home-symbolic")
            .tooltip_text("Home (H)")
            .css_classes(["toolbar-btn"])
            .build();
        btn_home.connect_clicked(|_| send_key('h'));

        let btn_lock = ToggleButton::builder()
            .icon_name("changes-prevent-symbolic")
            .tooltip_text("Lock (Space)")
            .css_classes(["toolbar-btn"])
            .build();
        btn_lock.connect_toggled(|b| {
            send_key(' ');
            b.set_icon_name(if b.is_active() {
                "channel-secure-symbolic"
            } else {
                "channel-insecure-symbolic"
            });
        });

        let btn_rec = ToggleButton::builder()
            .icon_name("media-record-symbolic")
            .tooltip_text("Record (R)")
            .css_classes(["toolbar-btn"])
            .build();
        btn_rec.connect_toggled(|_| send_key('r'));

        // ── Bottom info bar ──
        let info_l = Label::builder().css_classes(["info-label"]).label("").halign(Align::Start).hexpand(true).build();

        // ── Menu ──
        let show_osd_cb = Rc::new(RefCell::new(
            gtk4::CheckButton::builder().label("Show Overlay").active(true).build(),
        ));
        {
            let so = Arc::clone(&show_osd);
            show_osd_cb
                .borrow()
                .connect_toggled(move |cb| so.store(cb.is_active(), Ordering::Relaxed));
        }

        let quit_btn = Button::builder()
            .label("Quit")
            .icon_name("application-exit-symbolic")
            .halign(Align::Fill)
            .build();
        {
            let app = app.clone();
            quit_btn.connect_clicked(move |_| app.quit());
        }

        let menu_box = Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .margin_start(8).margin_end(8).margin_top(8).margin_bottom(8)
            .spacing(4)
            .build();
        menu_box.append(&*show_osd_cb.borrow());
        menu_box.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        let sl = Label::builder().label("  O: toggle overlay").margin_top(4).build();
        menu_box.append(&sl);
        menu_box.append(&quit_btn);

        let popover = gtk4::Popover::builder().child(&menu_box).build();
        let menu_btn = gtk4::MenuButton::builder()
            .icon_name("open-menu-symbolic")
            .tooltip_text("Menu")
            .popover(&popover)
            .build();

        let end_box = Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(2)
            .build();
        end_box.append(&btn_home);
        end_box.append(&btn_lock);
        end_box.append(&btn_rec);
        end_box.append(&menu_btn);

        // ── Layout ──
        let title_label = Label::builder().label("NeuraCam").build();
        let header = HeaderBar::builder().title_widget(&title_label).build();
        header.pack_end(&end_box);

        let main_box = Box::builder().orientation(gtk4::Orientation::Vertical).build();
        main_box.append(&header);
        main_box.append(&overlay);
        main_box.append(&info_l);

        let win = ApplicationWindow::builder()
            .application(app)
            .default_width(960)
            .default_height(640)
            .content(&main_box)
            .build();
        win.present();

        // ── Keyboard ──
        let kc = EventControllerKey::new();
        {
            let s_osd = Arc::clone(&show_osd);
            let osd_cb = Rc::clone(&show_osd_cb);
            kc.connect_key_pressed(move |_, keyval, _, _| {
                match keyval.name().as_deref() {
                    Some("q") => send_key('q'),
                    Some("h") => send_key('h'),
                    Some("space") => send_key(' '),
                    Some("r") => send_key('r'),
                    Some("o") => {
                        let new = !s_osd.load(Ordering::Relaxed);
                        s_osd.store(new, Ordering::Relaxed);
                        osd_cb.borrow().set_active(new);
                    }
                    _ => return gtk4::glib::Propagation::Proceed,
                }
                gtk4::glib::Propagation::Stop
            });
        }
        win.add_controller(kc);

        // ── IPC ──
        let (tx, rx) = mpsc::channel::<(Vec<u8>, NeuraCamState)>();
        let connected = Arc::new(AtomicBool::new(false));
        let c2 = Arc::clone(&connected);

        thread::spawn(move || loop {
            match UnixStream::connect(SOCKET_PATH) {
                Ok(mut stream) => {
                    c2.store(true, Ordering::Relaxed);
                    let mut frame: Option<Vec<u8>> = None;
                    let mut state: Option<NeuraCamState> = None;
                    loop {
                        match read_msg(&mut stream) {
                            Ok((t, p)) => {
                                match t {
                                    MSG_STATE => {
                                        if let Ok(s) = serde_json::from_slice(&p) {
                                            state = Some(s);
                                        }
                                    }
                                    MSG_FRAME => {
                                        frame = Some(p);
                                    }
                                    _ => {}
                                }
                                if let (Some(f), Some(s)) = (frame.take(), state.take()) {
                                    let _ = tx.send((f, s));
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }
                Err(_) => c2.store(false, Ordering::Relaxed),
            }
            thread::sleep(RECONNECT_DELAY);
        });

        // ── Update loop ──
        let c3 = Arc::clone(&connected);
        let mut frame_w = 1280.0_f64;
        let mut frame_h = 720.0_f64;
        gtk4::glib::timeout_add_local(Duration::from_millis(33), move || {
            loop {
                match rx.try_recv() {
                    Ok((jpeg, st)) => {
                        if let Ok(pb) = Pixbuf::from_read(std::io::Cursor::new(jpeg)) {
                            frame_w = st.frame_w.max(1.0);
                            frame_h = st.frame_h.max(1.0);
                            picture.set_pixbuf(Some(&pb));
                        }
                        if c3.load(Ordering::Relaxed) {
                            let mc = mode_color(&st.mode);
                            let mt = if st.mode == "IDLE" || st.mode == "SEARCH" {
                                format!("[ {} ]", st.mode)
                            } else {
                                format!("[ {} ({}) ]", st.mode, st.tracking_target)
                            };
                            osd_mode.set_markup(&format!(
                                "<span color='{}'>{}</span>",
                                mc, mt
                            ));

                            let fc = if st.fps > 20.0 { "#cccccc" }
                                else if st.fps > 10.0 { "#d4a040" }
                                else { "#e05555" };
                            osd_fps.set_markup(&format!(
                                "<span color='{}'>{:.1} fps</span>",
                                fc, st.fps
                            ));

                            let mut info_parts: Vec<String> = Vec::new();
                            if st.recording {
                                info_parts.push("<span color='#e05555'>● REC</span>".into());
                            }
                            if !st.serial_connected {
                                info_parts.push("<span color='#d4a040'>⚠ SRL</span>".into());
                            }
                            if st.face_detected {
                                info_parts.push(format!(
                                    "<span color='#cccccc'>FCE {:.2}</span>",
                                    st.face_confidence
                                ));
                            }
                            if st.gesture != "NONE" {
                                info_parts.push(format!(
                                    "<span color='#d4a040'>GES {}</span>",
                                    st.gesture
                                ));
                            }
                            if st.zoom_level > 1.0 {
                                info_parts.push(format!(
                                    "<span color='#d4a040'>Z {:.1}X</span>",
                                    st.zoom_level
                                ));
                            }
                            osd_info.set_markup(&info_parts.join("  "));
                            osd_info.set_visible(!info_parts.is_empty());

                            draw_face_overlay(&face_da, &st, frame_w, frame_h);
                        } else {
                            osd_mode
                                .set_markup("<span color='#e05555'>[ DISCONNECTED ]</span>");
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        osd_mode
                            .set_markup("<span color='#e05555'>[ IPC ERROR ]</span>");
                        break;
                    }
                }
            }
            gtk4::glib::ControlFlow::Continue
        });
    });

    app.run();
}
