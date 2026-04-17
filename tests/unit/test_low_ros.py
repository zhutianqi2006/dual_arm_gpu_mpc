from __future__ import annotations

from dual_arm_gpu_mpc.ros import low as low_ros_module


def test_low_ros_run_swallows_external_shutdown(monkeypatch):
    destroyed = {"value": False}

    class _FakeExternalShutdownException(Exception):
        pass

    def _fake_spin(node):
        raise _FakeExternalShutdownException()

    def _fake_destroy_node():
        destroyed["value"] = True

    fake_node = object.__new__(low_ros_module.LowROSModule)

    monkeypatch.setattr(low_ros_module, "ExternalShutdownException", _FakeExternalShutdownException, raising=False)
    monkeypatch.setattr(low_ros_module.rclpy, "spin", _fake_spin)
    monkeypatch.setattr(fake_node, "destroy_node", _fake_destroy_node)

    low_ros_module.LowROSModule.run(fake_node)

    assert destroyed["value"]
