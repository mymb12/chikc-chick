import time

import adafruit_mlx90640
import board
import busio
import matplotlib.pyplot as plt
import numpy as np

# Initialize I2C and MLX90640
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)  # 400kHz I2C
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ  # Lower = more stable

frame = [0] * 768  # MLX90640 is 32x24 = 768 pixels

while True:
    try:
        mlx.getFrame(frame)
        img = np.reshape(frame, (24, 32))

        # Display the thermal image
        plt.imshow(img, cmap="inferno")
        plt.colorbar()
        plt.pause(0.1)  # brief pause to update
        plt.clf()
    except ValueError:
        # Sometimes it throws sync errors; just retry
        continue
