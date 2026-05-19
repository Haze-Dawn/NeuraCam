import json
import os
import shutil
from src.main import ExperimentLogger


def test_logger_init():
    test_dir = "/tmp/test_logger_init"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    logger = ExperimentLogger(log_dir=test_dir)
    assert len(logger.entries) == 0
    shutil.rmtree(test_dir)


def test_logger_log():
    test_dir = "/tmp/test_logger_log"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    logger = ExperimentLogger(log_dir=test_dir)
    entry = {"frame": 1, "mode": "TRACKING", "fps": 30.0}
    logger.log(entry)
    assert len(logger.entries) == 1
    assert logger.entries[0] == entry
    shutil.rmtree(test_dir)


def test_logger_save():
    test_dir = "/tmp/test_logger_save"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    logger = ExperimentLogger(log_dir=test_dir)
    logger.log({"frame": 1})
    logger.log({"frame": 2})
    logger.save("test")
    files = os.listdir(test_dir)
    json_files = [f for f in files if f.startswith("test_") and f.endswith(".json")]
    assert len(json_files) == 1
    with open(os.path.join(test_dir, json_files[0])) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["frame"] == 1
    assert data[1]["frame"] == 2
    shutil.rmtree(test_dir)


def test_logger_save_empty():
    test_dir = "/tmp/test_logger_empty"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    logger = ExperimentLogger(log_dir=test_dir)
    logger.save("empty")
    files = os.listdir(test_dir)
    json_files = [f for f in files if f.endswith(".json")]
    assert len(json_files) == 1
    with open(os.path.join(test_dir, json_files[0])) as f:
        data = json.load(f)
    assert len(data) == 0
    shutil.rmtree(test_dir)


def test_logger_multiple_entries():
    test_dir = "/tmp/test_logger_multi"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    logger = ExperimentLogger(log_dir=test_dir)
    for i in range(10):
        logger.log({"frame": i, "value": i * 2})
    assert len(logger.entries) == 10
    assert logger.entries[-1]["frame"] == 9
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_logger_init()
    test_logger_log()
    test_logger_save()
    test_logger_save_empty()
    test_logger_multiple_entries()
    print("\nAll logger tests passed!")
