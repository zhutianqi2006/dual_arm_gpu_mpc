from __future__ import annotations

from pathlib import Path
import importlib.util
import fcntl
import os
import pty
import py_compile
import select
import signal
import subprocess
import sys
import termios
import time
from textwrap import dedent

import pytest


def _find_ancestor_with(start: Path, filename: str) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / filename).exists():
            return candidate
    raise AssertionError(f"Could not find {filename} above {start}")


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_ancestor_with(TEST_DIR, "pyproject.toml")

_DIRECT_RUN_IMPORT_CHECK = dedent(
    """
    from pathlib import Path
    import runpy
    import sys

    script = Path(sys.argv[1]).resolve()
    project_root = Path.cwd().resolve()
    src_root = (project_root / "src").resolve()

    trimmed = []
    for entry in sys.path:
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if resolved in {project_root, src_root}:
            continue
        trimmed.append(entry)

    sys.path[:] = [str(script.parent)] + trimmed
    runpy.run_path(str(script), run_name="example_import_test")
    """
)

_DIRECT_RUN_IMPORT_WITH_FAKE_DQ_TORCH = dedent(
    """
    from pathlib import Path
    import runpy
    import sys

    fake_root = Path(sys.argv[1]).resolve()
    script = Path(sys.argv[2]).resolve()
    project_root = Path.cwd().resolve()
    src_root = (project_root / "src").resolve()

    sys.path[:] = [str(fake_root), str(script.parent)] + [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() not in {project_root, src_root}
    ]
    runpy.run_path(str(script), run_name="example_import_test")
    """
)


@pytest.mark.parametrize(
    "relative_script",
    [
        "examples/sim1_ur/mppi_kmeans_adpan.py",
        "examples/sim1_ur/low_level.py",
        "examples/sim2_ur/mppi_kmeans_adpan.py",
        "examples/sim2_ur/low_level.py",
        "examples/sim3_ur/mppi_kmeans_adpan.py",
        "examples/sim3_ur/low_level.py",
        "examples/exp1_ur/mppi_kmeans_adpan.py",
        "examples/exp1_ur/low_level.py",
        "examples/sim1_ur/bullet_robot_ros.py",
        "examples/sim2_ur/bullet_robot_ros.py",
        "examples/sim3_ur/bullet_robot_ros.py",
        "examples/exp1_ur/bullet_robot_ros.py",
    ],
)
def test_examples_support_direct_script_imports(relative_script: str):
    script = PROJECT_ROOT / relative_script
    result = subprocess.run(
        [sys.executable, "-c", _DIRECT_RUN_IMPORT_CHECK, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Direct script import failed for {relative_script}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_examples_compile_cleanly():
    example_files = sorted((PROJECT_ROOT / "examples").rglob("*.py"))
    assert example_files
    for example_file in example_files:
        py_compile.compile(str(example_file), doraise=True)


def test_examples_only_keep_supported_kmeans_controller_entrypoints():
    expected = {
        "examples/exp1_ur/bullet_robot_ros.py",
        "examples/exp1_ur/low_level.py",
        "examples/exp1_ur/mppi_kmeans_adpan.py",
        "examples/exp1_ur/ur3_and_ur3e.yaml",
        "examples/sim1_ur/bullet_robot_ros.py",
        "examples/sim1_ur/low_level.py",
        "examples/sim1_ur/mppi_kmeans_adpan.py",
        "examples/sim1_ur/ur3_and_ur3e.yaml",
        "examples/sim2_ur/bullet_robot_ros.py",
        "examples/sim2_ur/low_level.py",
        "examples/sim2_ur/mppi_kmeans_adpan.py",
        "examples/sim2_ur/ur3_and_ur3e.yaml",
        "examples/sim3_ur/bullet_robot_ros.py",
        "examples/sim3_ur/low_level.py",
        "examples/sim3_ur/mppi_kmeans_adpan.py",
        "examples/sim3_ur/ur3_and_ur3e.yaml",
    }
    actual = {
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "examples").glob("*_*/*")
        if path.is_file() and path.parent.name in {"sim1_ur", "sim2_ur", "sim3_ur", "exp1_ur"}
    }
    assert actual == expected


def test_kmeans_examples_import_new_src_modules():
    for relative_script in (
        "examples/sim1_ur/mppi_kmeans_adpan.py",
        "examples/sim2_ur/mppi_kmeans_adpan.py",
        "examples/sim3_ur/mppi_kmeans_adpan.py",
        "examples/exp1_ur/mppi_kmeans_adpan.py",
    ):
        source = (PROJECT_ROOT / relative_script).read_text(encoding="utf-8")
        assert "from dual_arm_gpu_mpc.config.loader import ConfigModule" in source
        assert (
            "from dual_arm_gpu_mpc.controllers.high_level.mppi.kmeans_adpan import MPPIKmeansAdpAnModule"
            in source
        )


def test_low_level_examples_import_new_src_modules():
    for relative_script in (
        "examples/sim1_ur/low_level.py",
        "examples/sim2_ur/low_level.py",
        "examples/sim3_ur/low_level.py",
        "examples/exp1_ur/low_level.py",
    ):
        source = (PROJECT_ROOT / relative_script).read_text(encoding="utf-8")
        assert "from dual_arm_gpu_mpc.config.loader import ConfigModule" in source
        assert (
            "from dual_arm_gpu_mpc.controllers.low_level.relative import RelativePoseLowLevelController"
            in source
        )


def test_low_level_examples_use_legacy_ur_targets():
    for relative_script in (
        "examples/sim1_ur/low_level.py",
        "examples/sim2_ur/low_level.py",
        "examples/sim3_ur/low_level.py",
        "examples/exp1_ur/low_level.py",
    ):
        source = (PROJECT_ROOT / relative_script).read_text(encoding="utf-8")
        assert "from dual_arm_gpu_mpc.common.example_presets import LEGACY_UR_LOW_LEVEL_TARGETS" in source
        assert "desire_abs_pose = [0.055365" not in source
        assert "desire_rel_pose = [0.043815" not in source


def test_kept_examples_use_new_src_bootstrap():
    for relative_script in (
        "examples/sim1_ur/mppi_kmeans_adpan.py",
        "examples/sim1_ur/low_level.py",
        "examples/sim2_ur/mppi_kmeans_adpan.py",
        "examples/sim2_ur/low_level.py",
        "examples/sim3_ur/mppi_kmeans_adpan.py",
        "examples/sim3_ur/low_level.py",
        "examples/exp1_ur/mppi_kmeans_adpan.py",
        "examples/exp1_ur/low_level.py",
        "examples/sim1_ur/bullet_robot_ros.py",
        "examples/sim2_ur/bullet_robot_ros.py",
        "examples/sim3_ur/bullet_robot_ros.py",
        "examples/exp1_ur/bullet_robot_ros.py",
    ):
        source = (PROJECT_ROOT / relative_script).read_text(encoding="utf-8")
        assert "from dual_arm_gpu_mpc.common.example_bootstrap import bootstrap_example_paths" in source


def test_bullet_examples_use_internal_pybullet_helpers():
    for relative_script in (
        "examples/sim1_ur/bullet_robot_ros.py",
        "examples/sim2_ur/bullet_robot_ros.py",
        "examples/sim3_ur/bullet_robot_ros.py",
        "examples/exp1_ur/bullet_robot_ros.py",
    ):
        source = (PROJECT_ROOT / relative_script).read_text(encoding="utf-8")
        assert "BulletJointResetRobot" in source


def test_sim3_bullet_example_contains_dynamic_obstacle_support():
    source = (PROJECT_ROOT / "examples" / "sim3_ur" / "bullet_robot_ros.py").read_text(encoding="utf-8")
    assert "PingPongObstacleTrajectory" in source
    assert '"/dock_position_world"' in source
    assert "moving_obstacle" in source


def test_standalone_examples_use_new_src_bootstrap():
    for relative_script in (
        "examples/compute_dq_base.py",
        "examples/abs_pose_translation.py",
        "examples/ur_real_arm_ros.py",
        "examples/ur_vel_reset_ros.py",
    ):
        source = (PROJECT_ROOT / relative_script).read_text(encoding="utf-8")
        assert "from dual_arm_gpu_mpc.common.example_bootstrap import bootstrap_example_paths" in source


def test_real_arm_example_uses_internal_pybullet_wrapper():
    source = (PROJECT_ROOT / "examples" / "ur_real_arm_ros.py").read_text(encoding="utf-8")
    assert "from dual_arm_gpu_mpc.common.pybullet_robot import BulletJointResetRobot" in source


def test_removed_legacy_examples_are_absent():
    assert not (PROJECT_ROOT / "examples" / "ur_acc_test.py").exists()
    assert not (PROJECT_ROOT / "examples" / "real_world_tf_change.py").exists()


def test_legacy_helper_directory_is_removed():
    assert not (PROJECT_ROOT / ("u" + "tils")).exists()


def test_example_bootstrap_resolves_legacy_bullet_urdf_path():
    bootstrap_path = PROJECT_ROOT / "src" / "dual_arm_gpu_mpc" / "common" / "example_bootstrap.py"
    spec = importlib.util.spec_from_file_location("example_bootstrap_for_tests", bootstrap_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    resolved = module.resolve_legacy_example_path("model/dual_arm_model/dual_arm_model.urdf")

    assert isinstance(resolved, str)
    assert resolved.startswith("/")
    assert Path(resolved).exists()


def test_kmeans_example_bootstrap_prefers_workspace_dq_torch(tmp_path: Path):
    fake_root = tmp_path / "fake_site"
    fake_root.mkdir()
    (fake_root / "dq_torch.py").write_text(
        "raise ImportError('fake dq_torch from external site-packages was imported')\n",
        encoding="utf-8",
    )

    script = PROJECT_ROOT / "examples" / "sim1_ur" / "mppi_kmeans_adpan.py"
    result = subprocess.run(
        [sys.executable, "-c", _DIRECT_RUN_IMPORT_WITH_FAKE_DQ_TORCH, str(fake_root), str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Example import should prefer workspace dq_torch over external dq_torch.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_example_bootstrap_exposes_project_venv_site_packages_to_system_python():
    check = dedent(
        """
        import importlib.util
        import sys
        from pathlib import Path

        project_root = Path(sys.argv[1]).resolve()
        bootstrap_path = project_root / "src" / "dual_arm_gpu_mpc" / "common" / "example_bootstrap.py"
        spec = importlib.util.spec_from_file_location("example_bootstrap_for_system_python", bootstrap_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None
        assert spec.loader is not None
        spec.loader.exec_module(module)

        module.bootstrap_example_paths(project_root / "examples" / "sim1_ur" / "mppi_kmeans_adpan.py")

        import importlib.util as iu

        missing = [name for name in ("kmeans_pytorch", "dqrobotics") if iu.find_spec(name) is None]
        raise SystemExit(0 if not missing else 1)
        """
    )

    result = subprocess.run(
        ["/usr/bin/python3", "-c", check, str(PROJECT_ROOT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "System python should see project .venv dependencies after bootstrap.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def _load_example_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_example_in_pty_and_send_ctrl_c(
    relative_script: str,
    *,
    startup_seconds: float = 5.0,
    settle_timeout_seconds: float = 8.0,
):
    master_fd, slave_fd = pty.openpty()

    def _claim_controlling_terminal():
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

    process = subprocess.Popen(
        ["/usr/bin/python3", str(PROJECT_ROOT / relative_script)],
        cwd=str(PROJECT_ROOT),
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        preexec_fn=_claim_controlling_terminal,
    )
    os.close(slave_fd)

    captured_chunks: list[bytes] = []
    try:
        time.sleep(startup_seconds)
        os.write(master_fd, b"\x03")

        deadline = time.time() + settle_timeout_seconds
        while time.time() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master_fd, 65536)
                except OSError:
                    chunk = b""
                if chunk:
                    captured_chunks.append(chunk)

            if process.poll() is not None:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break
                    try:
                        chunk = os.read(master_fd, 65536)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        break
                    captured_chunks.append(chunk)
                break
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        try:
            os.close(master_fd)
        except OSError:
            pass

    return process.returncode, b"".join(captured_chunks).decode("utf-8", "replace")


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("sim1_bullet_interrupt_test", "examples/sim1_ur/bullet_robot_ros.py"),
        ("sim2_bullet_interrupt_test", "examples/sim2_ur/bullet_robot_ros.py"),
        ("exp1_bullet_interrupt_test", "examples/exp1_ur/bullet_robot_ros.py"),
    ],
)
def test_bullet_examples_swallow_keyboard_interrupt_during_initialization(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
):
    module = _load_example_module(module_name, relative_path)

    class _FakeRclpy:
        def __init__(self):
            self.shutdown_called = False

        def init(self, args=None):
            return None

        def ok(self):
            return True

        def shutdown(self):
            self.shutdown_called = True

    fake_rclpy = _FakeRclpy()

    def _raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "rclpy", fake_rclpy)
    monkeypatch.setattr(module, "DualArmBulletModel", _raise_interrupt)
    monkeypatch.setattr(module, "_exit_after_interrupt", lambda exit_code=130: None)

    module.main()

    assert fake_rclpy.shutdown_called


def test_bullet_example_exits_promptly_after_terminal_ctrl_c():
    returncode, output = _run_example_in_pty_and_send_ctrl_c("examples/sim1_ur/bullet_robot_ros.py")

    assert returncode is not None, (
        "Bullet example should exit after terminal Ctrl+C instead of hanging.\n"
        f"OUTPUT:\n{output}"
    )
    assert returncode in {0, 130, -signal.SIGINT}, (
        "Bullet example should exit cleanly after terminal Ctrl+C.\n"
        f"RETURNCODE: {returncode}\n"
        f"OUTPUT:\n{output}"
    )
    assert "Segmentation fault" not in output, output


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("sim1_kmeans_interrupt_test", "examples/sim1_ur/mppi_kmeans_adpan.py"),
        ("sim2_kmeans_interrupt_test", "examples/sim2_ur/mppi_kmeans_adpan.py"),
        ("exp1_kmeans_interrupt_test", "examples/exp1_ur/mppi_kmeans_adpan.py"),
    ],
)
def test_kmeans_examples_swallow_keyboard_interrupt_and_shutdown_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
):
    module = _load_example_module(module_name, relative_path)

    class _FakeRclpy:
        def __init__(self):
            self.shutdown_called = False

        def init(self, args=None):
            return None

        def ok(self):
            return True

        def shutdown(self):
            self.shutdown_called = True

    class _FakeModule:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def warm_up(self):
            raise KeyboardInterrupt

        def shutdown(self):
            self.closed = True
            fake_rclpy.shutdown()

    fake_rclpy = _FakeRclpy()
    fake_instance = _FakeModule()

    def _fake_ctor(*args, **kwargs):
        return fake_instance

    monkeypatch.setattr(module, "rclpy", fake_rclpy)
    monkeypatch.setattr(module, "MPPIKmeansAdpAnModule", _fake_ctor)

    module.main()

    assert fake_instance.closed
    assert fake_rclpy.shutdown_called


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("sim1_low_level_interrupt_test", "examples/sim1_ur/low_level.py"),
        ("sim2_low_level_interrupt_test", "examples/sim2_ur/low_level.py"),
        ("exp1_low_level_interrupt_test", "examples/exp1_ur/low_level.py"),
    ],
)
def test_low_level_examples_swallow_keyboard_interrupt_and_shutdown_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
):
    module = _load_example_module(module_name, relative_path)

    class _FakeRclpy:
        def __init__(self):
            self.shutdown_called = False

        def init(self, args=None):
            return None

        def ok(self):
            return True

        def shutdown(self):
            self.shutdown_called = True

    class _FakeController:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def play_once(self):
            raise KeyboardInterrupt

        def shutdown(self):
            self.closed = True
            fake_rclpy.shutdown()

    fake_rclpy = _FakeRclpy()
    fake_controller = _FakeController()

    monkeypatch.setattr(module, "rclpy", fake_rclpy)
    monkeypatch.setattr(module, "RelativePoseLowLevelController", lambda *args, **kwargs: fake_controller)

    module.main()

    assert fake_controller.closed
    assert fake_rclpy.shutdown_called
