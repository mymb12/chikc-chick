
import time

import numpy as np
import smbus2

# Define I2C address of the thermal camera (check the address via i2cdetect)
I2C_ADDRESS = 0x33

# Define registers for the camera's configuration (refer to the datasheet for details)
REG_TEMP = 0x80  # Register to read temperature data

# Initialize I2C bus
bus = smbus2.SMBus(1)  # 1 means /dev/i2c-1

def read_temperature_data():
    try:
        # Read raw data from the camera (adjust based on the camera's register specifications)
        data = bus.read_i2c_block_data(I2C_ADDRESS, REG_TEMP, 128)  # Example for 128 bytes
        temperatures = np.array(data)  # Convert data into a NumPy array for processing
        return temperatures
    except Exception as e:
        print(f"Error reading data from camera: {e}")
        return None

def process_temperature_data(data):
    # Example of processing the data into a 2D array (e.g., 32x24 thermal image)
    # Adjust the size based on your camera's resolution (check the datasheet)
    image_data = np.reshape(data, (24, 32))  # 24x32 grid for 32x24 resolution cameras
    return image_data

def display_image(image_data):
    # Example code for visualizing the thermal image using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(image_data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Main loop to read and display temperature data
while True:
    raw_data = read_temperature_data()
    if raw_data is not None:
        image_data = process_temperature_data(raw_data)
        display_image(image_data)
    time.sleep(1)  # Adjust the sleep time as necessary

