import numpy as np
import matplotlib.pyplot as plt

path = "logs/manip_trace_20260114_190115.npz"
d = np.load(path, allow_pickle=True)

keys = set(d.files)

rows = d["rows"].item() if np.asarray(d["rows"]).shape == () else d["rows"]
metric = d["metric"].item() if "metric" in keys and np.asarray(d["metric"]).shape == () else (d["metric"] if "metric" in keys else "")

if {"t", "manipulability"}.issubset(keys):
	# old schema
	t = d["t"]
	m = d["manipulability"]
	x = t
	xlabel = "time [s]"
	y = m
	title = f"manip trace ({rows}, {metric})"
elif {"wall_time", "manip_mean"}.issubset(keys):
	# new schema (kmeans module): per-call stats
	wall_time = d["wall_time"]
	x = wall_time - wall_time[0]
	xlabel = "wall time since start [s]"
	y = d["manip_mean"]
	y_min = d["manip_min"] if "manip_min" in keys else None
	y_max = d["manip_max"] if "manip_max" in keys else None
	title = f"manip trace mean ({rows}, {metric})"
else:
	raise KeyError(f"Unknown log schema. Available keys: {sorted(d.files)}")

plt.figure()
plt.plot(x, y, lw=1.5, label="mean" if "manip_mean" in keys else None)
if "wall_time" in keys and "manip_mean" in keys and y_min is not None and y_max is not None:
	plt.fill_between(x, y_min, y_max, alpha=0.2, label="min-max")
plt.xlabel(xlabel)
plt.ylabel("manipulability")
plt.title(title)
if "wall_time" in keys and "manip_mean" in keys:
	plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()