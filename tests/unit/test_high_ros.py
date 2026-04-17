from __future__ import annotations

from dual_arm_gpu_mpc.ros import high as high_ros_module


def test_high_ros_run_swallows_external_shutdown(monkeypatch):
    destroyed = {"value": False}

    class _FakeExternalShutdownException(Exception):
        pass

    def _fake_spin(node):
        raise _FakeExternalShutdownException()

    def _fake_destroy_node():
        destroyed["value"] = True

    fake_node = object.__new__(high_ros_module.HighROSModule)

    monkeypatch.setattr(high_ros_module, "ExternalShutdownException", _FakeExternalShutdownException, raising=False)
    monkeypatch.setattr(high_ros_module.rclpy, "spin", _fake_spin)
    monkeypatch.setattr(fake_node, "destroy_node", _fake_destroy_node)

    high_ros_module.HighROSModule.run(fake_node)

    assert destroyed["value"]
