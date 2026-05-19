from src.control.state_machine import StateMachine, Mode


def test_sm_init():
    sm = StateMachine()
    assert sm.mode == Mode.IDLE
    assert sm.gesture_hold_frames == 5
    assert sm.idle_timeout == 150
    assert sm.search_progress == 0.0
    assert sm.gesture_hold_progress == 0.0


def test_sm_idle_to_tracking():
    sm = StateMachine()
    sm.update_face_status(True)
    assert sm.mode == Mode.TRACKING


def test_sm_tracking_to_search():
    sm = StateMachine(idle_timeout_frames=3)
    sm.update_face_status(True)
    assert sm.mode == Mode.TRACKING
    for _ in range(10):
        sm.update_face_status(False)
    assert sm.mode == Mode.SEARCH


def test_sm_search_to_idle():
    sm = StateMachine(search_duration=3)
    sm.mode = Mode.SEARCH
    for _ in range(5):
        sm.update_face_status(False)
    assert sm.mode == Mode.IDLE


def test_sm_search_to_tracking():
    sm = StateMachine(search_duration=10, idle_timeout_frames=2)
    sm.update_face_status(True)
    for _ in range(5):
        sm.update_face_status(False)
    assert sm.mode == Mode.SEARCH
    sm.update_face_status(True)
    assert sm.mode == Mode.TRACKING


def test_sm_tracking_locked_open_palm():
    sm = StateMachine(gesture_hold_frames=1)
    sm.update_face_status(True)
    assert sm.mode == Mode.TRACKING
    sm.process_gesture("OPEN_PALM")
    assert sm.mode == Mode.LOCKED


def test_sm_locked_tracking_fist():
    sm = StateMachine(gesture_hold_frames=1)
    sm.mode = Mode.LOCKED
    sm.process_gesture("FIST")
    assert sm.mode == Mode.TRACKING


def test_sm_thumbs_up_home():
    sm = StateMachine(gesture_hold_frames=1)
    sm.update_face_status(True)
    sm.process_gesture("THUMBS_UP")
    assert sm.mode == Mode.HOME


def test_sm_gesture_hold_required():
    sm = StateMachine(gesture_hold_frames=3)
    sm.update_face_status(True)
    sm.process_gesture("OPEN_PALM")
    assert sm.mode == Mode.TRACKING
    sm.process_gesture("OPEN_PALM")
    assert sm.mode == Mode.TRACKING
    sm.process_gesture("OPEN_PALM")
    assert sm.mode == Mode.LOCKED


def test_sm_gesture_hold_resets_on_change():
    sm = StateMachine(gesture_hold_frames=3)
    sm.update_face_status(True)
    sm.process_gesture("OPEN_PALM")
    sm.process_gesture("OPEN_PALM")
    sm.process_gesture("FIST")
    assert sm.mode == Mode.TRACKING
    sm.process_gesture("FIST")
    sm.process_gesture("FIST")
    assert sm.mode == Mode.TRACKING
    sm.process_gesture("OPEN_PALM")
    sm.process_gesture("OPEN_PALM")
    sm.process_gesture("OPEN_PALM")
    assert sm.mode == Mode.LOCKED


def test_sm_gesture_none_ignored():
    sm = StateMachine(gesture_hold_frames=1)
    sm.update_face_status(True)
    sm.process_gesture("NONE")
    assert sm.mode == Mode.TRACKING


def test_sm_finish_homing():
    sm = StateMachine()
    sm.mode = Mode.HOME
    sm.finish_homing()
    assert sm.mode == Mode.TRACKING


def test_sm_toggle_lock():
    sm = StateMachine()
    sm.update_face_status(True)
    sm.toggle_lock()
    assert sm.mode == Mode.LOCKED
    sm.toggle_lock()
    assert sm.mode == Mode.TRACKING


def test_sm_search_progress():
    sm = StateMachine(search_duration=10)
    assert sm.search_progress == 0.0
    sm.mode = Mode.SEARCH
    for i in range(5):
        sm.update_face_status(False)
    assert sm.search_progress == 0.5
    for i in range(5):
        sm.update_face_status(False)
    assert sm.search_progress == 1.0


def test_sm_gesture_hold_progress():
    sm = StateMachine(gesture_hold_frames=4)
    assert sm.gesture_hold_progress == 0.0
    sm._gesture_hold_counter = 2
    assert sm.gesture_hold_progress == 0.5
    sm._gesture_hold_counter = 4
    assert sm.gesture_hold_progress == 1.0


def test_sm_search_active_property():
    sm = StateMachine()
    assert sm.search_active is False
    sm.mode = Mode.SEARCH
    assert sm.search_active is True


def test_sm_wave_single_toggles_hand_tracking():
    sm = StateMachine()
    sm.mode = Mode.TRACKING
    assert sm.tracking_target == "FACE"
    assert sm.zoom_mode_active is False

    sm.process_wave('single')
    assert sm.mode == Mode.TRACKING_HAND
    assert sm.tracking_target == "HAND"
    assert sm.zoom_mode_active is True

    sm.process_wave('single')
    assert sm.mode == Mode.TRACKING
    assert sm.tracking_target == "FACE"
    assert sm.zoom_mode_active is False
    assert sm.zoom_level == 1.0


def test_sm_wave_single_only_in_tracking_or_hand():
    sm = StateMachine()
    sm.mode = Mode.IDLE
    sm.process_wave('single')
    assert sm.mode == Mode.IDLE

    sm.mode = Mode.LOCKED
    sm.process_wave('single')
    assert sm.mode == Mode.LOCKED


def test_sm_wave_double_toggles_zoom_in_hand_mode():
    sm = StateMachine()
    sm.mode = Mode.TRACKING_HAND
    sm.zoom_mode_active = True
    sm.zoom_level = 2.0

    sm.process_wave('double')
    assert sm.zoom_mode_active is False
    assert sm.zoom_level == 1.0

    sm.process_wave('double')
    assert sm.zoom_mode_active is True
    assert sm.zoom_level == 1.0


def test_sm_wave_double_ignored_outside_hand_mode():
    sm = StateMachine()
    sm.mode = Mode.TRACKING
    sm.zoom_mode_active = True
    sm.process_wave('double')
    assert sm.zoom_mode_active is True


def test_sm_hand_zoom_mapping():
    sm = StateMachine()
    sm.zoom_mode_active = True
    sm.update_hand_zoom(4)
    assert sm.zoom_level == 1.0
    sm.update_hand_zoom(3)
    assert sm.zoom_level == 1.3
    sm.update_hand_zoom(2)
    assert sm.zoom_level == 1.7
    sm.update_hand_zoom(1)
    assert sm.zoom_level == 2.5
    sm.update_hand_zoom(0)
    assert sm.zoom_level == 3.0


def test_sm_hand_zoom_ignored_when_inactive():
    sm = StateMachine()
    sm.zoom_mode_active = False
    sm.update_hand_zoom(0)
    assert sm.zoom_level == 1.0


def test_sm_locked_fist_returns_to_pre_lock_mode():
    sm = StateMachine(gesture_hold_frames=1)
    sm.mode = Mode.TRACKING_HAND
    sm.process_gesture("OPEN_PALM")
    assert sm.mode == Mode.LOCKED
    sm.process_gesture("FIST")
    assert sm.mode == Mode.TRACKING_HAND


def test_sm_toggle_lock_preserves_pre_lock_mode():
    sm = StateMachine()
    sm.mode = Mode.TRACKING_HAND
    sm.toggle_lock()
    assert sm.mode == Mode.LOCKED
    sm.toggle_lock()
    assert sm.mode == Mode.TRACKING_HAND


def test_sm_open_palm_blocks_in_hand_track():
    sm = StateMachine(gesture_hold_frames=1)
    sm.mode = Mode.TRACKING_HAND
    sm.process_gesture("OPEN_PALM")
    assert sm.mode == Mode.LOCKED


def test_sm_pointe_peace_no_longer_trigger_actions():
    sm = StateMachine(gesture_hold_frames=1)
    sm.mode = Mode.TRACKING
    sm.process_gesture("POINT")
    assert sm.mode == Mode.TRACKING
    sm.process_gesture("PEACE")
    assert sm.zoom_mode_active is False


if __name__ == "__main__":
    test_sm_init()
    test_sm_idle_to_tracking()
    test_sm_tracking_to_search()
    test_sm_search_to_idle()
    test_sm_search_to_tracking()
    test_sm_tracking_locked_open_palm()
    test_sm_locked_tracking_fist()
    test_sm_thumbs_up_home()
    test_sm_gesture_hold_required()
    test_sm_gesture_hold_resets_on_change()
    test_sm_gesture_none_ignored()
    test_sm_finish_homing()
    test_sm_toggle_lock()
    test_sm_search_progress()
    test_sm_gesture_hold_progress()
    test_sm_search_active_property()
    test_sm_wave_single_toggles_hand_tracking()
    test_sm_wave_single_only_in_tracking_or_hand()
    test_sm_wave_double_toggles_zoom_in_hand_mode()
    test_sm_wave_double_ignored_outside_hand_mode()
    test_sm_hand_zoom_mapping()
    test_sm_hand_zoom_ignored_when_inactive()
    test_sm_locked_fist_returns_to_pre_lock_mode()
    test_sm_toggle_lock_preserves_pre_lock_mode()
    test_sm_open_palm_blocks_in_hand_track()
    test_sm_pointe_peace_no_longer_trigger_actions()
    print("\nAll state machine tests passed!")
