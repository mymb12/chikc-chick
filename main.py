
import time
import numpy as np
import matplotlib.pyplot as plt
from mlx90640 import MLX90640

# Initialize MLX90640
sensor = MLX90640(i2c_dev="/dev/i2c-1", address=0x33)
sensor.refresh_rate = MLX90640.RefreshRate.REFRESH_8_HZ  # Can be 0.5, 1, 2, 4, 8, 16, 32, 64 Hz

frame = [0] * 768  # 32x24 = 768 pixels

plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
img = ax.imshow(np.zeros((24, 32)), cmap='inferno', vmin=20, vmax=40)
plt.colorbar(img)

try:
    while True:
        sensor.get_frame(frame)
        data = np.reshape(frame, (24, 32))
        img.set_data(data)
        plt.draw()
        plt.pause(0.001)

except KeyboardInterrupt:
    print("Stopped.")
