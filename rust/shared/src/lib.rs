use serde::Deserialize;
use std::collections::HashMap;
use std::io::{self, Read};

pub const MSG_STATE: u8 = 0;
pub const MSG_FRAME: u8 = 1;
pub const MSG_KEY: u8 = 2;

#[derive(Deserialize, Clone, Debug, Default)]
pub struct NeuraCamState {
    pub fps: f64,
    pub mode: String,
    pub tracking_target: String,
    pub face_detected: bool,
    pub face_x: f64,
    pub face_y: f64,
    pub face_w: f64,
    pub face_h: f64,
    pub face_confidence: f64,
    pub frame_w: f64,
    pub frame_h: f64,
    pub pan_angle: i64,
    pub tilt_angle: i64,
    pub pan_target: i64,
    pub tilt_target: i64,
    pub imu_pitch: f64,
    pub imu_roll: f64,
    pub imu_yaw: f64,
    pub kalman_uncertainty: f64,
    pub zoom_level: f64,
    pub gesture: String,
    pub gesture_method: String,
    pub recording: bool,
    pub serial_connected: bool,
    pub hand_detected: bool,
    pub pid_pan_error: f64,
    pub pid_tilt_error: f64,
    pub pid_pan_output: f64,
    pub pid_tilt_output: f64,
    pub pid_pan_p: f64,
    pub pid_pan_i: f64,
    pub pid_pan_d: f64,
    pub pid_tilt_p: f64,
    pub pid_tilt_i: f64,
    pub pid_tilt_d: f64,
    pub latency_ms: Option<HashMap<String, f64>>,
    pub events: Vec<String>,
    pub timestamp: f64,
}

pub fn read_msg(stream: &mut impl Read) -> io::Result<(u8, Vec<u8>)> {
    let mut header = [0u8; 5];
    stream.read_exact(&mut header)?;
    let len = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
    let msg_type = header[4];
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    Ok((msg_type, payload))
}
