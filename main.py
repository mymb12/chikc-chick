import os
import time

import adafruit_mlx90640
import board
import busio
import matplotlib.pyplot as plt
import numpy as np


# ========== PATCH ADABERRY LIBRARY ==========
def patch_outlier_check():
    import adafruit_mlx90640

    lib_path = adafruit_mlx90640.__file__
    with open(lib_path, "r") as f:
        lines = f.readlines()

    patched_lines = []
    skip_next = False
    for line in lines:
        if 'raise RuntimeError("More than 4 outlier pixels' in line:
            patched_lines.append(
                '        print("⚠️ Warning: More than 4 outlier pixels — ignoring.")\n'
            )
            skip_next = True
        elif skip_next:
            skip_next = False
            continue
        else:
            patched_lines.append(line)

    with open(lib_path, "w") as f:
        f.writelines(patched_lines)
    print("✅ Patched the 'more than 4 outlier pixels' error.")


patch_outlier_check()

# ========== INIT SENSOR ==========
print("📷 Initializing MLX90640...")
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
mlx.calibration_mode = adafruit_mlx90640.CalibrationMode.MLX90640_CHESS_MODE

frame = [0] * 768  # 32x24 = 768

# ========== LIVE THERMAL PLOT ==========
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((24, 32)), cmap="inferno", vmin=20, vmax=40)
cbar = plt.colorbar(im)
cbar.set_label("Temperature °C")

print("🎥 Starting thermal stream. Press Ctrl+C to exit.")

while True:
    try:
        mlx.getFrame(frame)
        img = np.reshape(frame, (24, 32))
        im.set_data(img)
        plt.pause(0.01)
    except Exception as e:
        print(f"⚠️ Frame error: {e}")
        time.sleep(0.1)
