import importlib


def test_package_imports_work_in_project_venv():
    project_pkg = importlib.import_module("dual_arm_gpu_mpc")
    dq_torch = importlib.import_module("dq_torch")
    curobo = importlib.import_module("curobo")

    assert project_pkg is not None
    assert hasattr(dq_torch, "mppi_project_step")
    assert hasattr(curobo, "__file__")
