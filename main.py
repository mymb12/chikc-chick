import time

import adafruit_mlx90640
import board
import busio
import matplotlib.pyplot as plt
import numpy as np

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

print("MLX90640 serial number:", [hex(i) for i in mlx.serial_number])

frame = [0] * 768

plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((24, 32)), cmap="inferno", vmin=20, vmax=40)
cbar = plt.colorbar(im)
cbar.set_label("Temperature °C")


while True:
    try:
        mlx.getFrame(frame)
        img = np.reshape(frame, (24, 32))
        im.set_data(img)
        plt.pause(0.01)

    except RuntimeError as e:
        if "More than 4 outlier pixels" in str(e):
            print("⚠️ Warning: More than 4 outlier pixels — ignoring.")
            continue
        else:
            print(f"⚠️ Runtime error: {e}")
            time.sleep(0.1)

    except OSError as e:
        print(f"❌ I2C communication error: {e}")
        time.sleep(0.5)

    except Exception as e:
        print(f"🔥 Unexpected error: {e}")
        time.sleep(0.5)
