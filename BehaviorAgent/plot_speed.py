'''This module plot the speed of the car based on the file speed.txt'''
import matplotlib.pyplot as plt
import numpy as np


def plot_speed():
    '''This function plot the speed of the car based on the file speed.txt'''
    # Read the data from the file
    with open('speed.txt', 'r') as f:
        lines = f.readlines()

    # Extract the time and speed values
    time = []
    speed = []
    reference_speed = []
    for line in lines:
        t, s, r_s = line.split(';')
        time.append(float(t))
        speed.append(float(s))
        reference_speed.append(float(r_s))


    # Convert to numpy arrays
    time = np.array(time)
    speed = np.array(speed)
    reference_speed = np.array(reference_speed)

    # Calculate the speed error
    speed_error = speed - reference_speed

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(time, speed, label='Speed', color='blue')
    plt.plot(time, reference_speed, label='Reference Speed', color='red')
    # plt.plot(time, speed_error, label='Speed Error', color='green')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.title('Speed of the car')
    plt.grid()
    plt.savefig('speed_plot.png')
if __name__ == '__main__':
    plot_speed()