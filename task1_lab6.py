from module_lab6 import *
import matplotlib.pyplot as plt

# Read in the production and injection well data
ti_pw1, yi_pw1 = np.genfromtxt('PW1.dat', delimiter=',', skip_header=1).T
ti_pw2, yi_pw2 = np.genfromtxt('PW2.dat', delimiter=',', skip_header=1).T
ti_iw1, yi_iw1 = np.genfromtxt('IW1.dat', delimiter=',', skip_header=1).T

# Define the measurement points for the linearly interpolated data
tj = np.arange(1980, 2015, 0.5)

# Interpolate the production and injection well data
yj_pw1 = interpolate_linear(ti_pw1, yi_pw1, tj, default=0)
yj_pw2 = interpolate_linear(ti_pw2, yi_pw2, tj, default=0)
yj_iw1 = interpolate_linear(ti_iw1, yi_iw1, tj, default=0)

# Constants
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

# Calculate the net rate of mass change at each point in time in kg per year
net_mass_change_rate = SECONDS_PER_YEAR * (yj_iw1 - yj_pw1 - yj_pw2)

# Integrate the net mass change over time to get the cumulative net mass change
cumulative_net_mass_change = [integrate_composite_trapezoid(tj[:i+1], net_mass_change_rate[:i+1]) for i in range(len(tj))]

# Plot the cumulative net mass change as a function of time
plt.figure(figsize=(10, 6))
plt.plot(tj, cumulative_net_mass_change, label='Cumulative net mass change')
plt.xlabel('Time (year)')
plt.ylabel('Cumulative net mass change')

# Add labels to indicate the earthquakes
earthquake_times = [2003.5, 2004.5, 2005]
earthquake_magnitudes = [3.5, 4.0, 4.3]
for i in range(len(earthquake_times)):
    plt.text(earthquake_times[i], 0, f'Earthquake {i+1}\nMagnitude: {earthquake_magnitudes[i]}')

plt.title('Cumulative net mass change over time')
plt.legend()
plt.grid(True)
plt.show()
