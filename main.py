import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Constants
G = 6.67430e-11
AU = 1.496e11


# Masses
M_sun = 1.989e30
M_earth = 5.972e24


# Initial Conditions
r_sun = np.array([0.0, 0.0])
r_earth = np.array([1.496e11, 0.0])

v_sun = np.array([0.0, 0.0])
v_earth = np.array([0.0, 27000.0])


# Time Setup
dt = 60 * 60 * 12
steps = 4000


# Storage
earth_positions = []
sun_positions = []
earth_speeds = []
energies = []


# Acceleration Function
def acceleration(r1, r2, m2):
    r = r2 - r1
    dist = np.linalg.norm(r)
    return G * m2 * r / dist**3

a_earth = acceleration(r_earth, r_sun, M_sun)
a_sun = acceleration(r_sun, r_earth, M_earth)


# Simulation (Verlet)
for _ in range(steps):

    r_earth += v_earth * dt + 0.5 * a_earth * dt**2
    r_sun += v_sun * dt + 0.5 * a_sun * dt**2

    new_a_earth = acceleration(r_earth, r_sun, M_sun)
    new_a_sun = acceleration(r_sun, r_earth, M_earth)

    v_earth += 0.5 * (a_earth + new_a_earth) * dt
    v_sun += 0.5 * (a_sun + new_a_sun) * dt

    a_earth = new_a_earth
    a_sun = new_a_sun

    earth_positions.append(r_earth.copy())
    sun_positions.append(r_sun.copy())
    earth_speeds.append(np.linalg.norm(v_earth))

    # Energy
    r = np.linalg.norm(r_earth - r_sun)
    KE = 0.5 * M_earth * np.linalg.norm(v_earth)**2 + \
         0.5 * M_sun * np.linalg.norm(v_sun)**2
    PE = -G * M_sun * M_earth / r
    energies.append([KE, PE, KE + PE])

earth_positions = np.array(earth_positions)
sun_positions = np.array(sun_positions)
earth_speeds = np.array(earth_speeds)
energies = np.array(energies)


# Barycenter Frame + AU
barycenter = (M_sun * sun_positions + M_earth * earth_positions) / (M_sun + M_earth)
earth_positions = (earth_positions - barycenter) / AU
sun_positions = (sun_positions - barycenter) / AU


# Plot Setup 
fig, (ax, ax_energy) = plt.subplots(1, 2, figsize=(14, 6))

# Dark theme
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax_energy.set_facecolor('black')

# Star field
np.random.seed(42)
stars_x = np.random.uniform(-2, 2, 200)
stars_y = np.random.uniform(-2, 2, 200)
ax.scatter(stars_x, stars_y, color='white', s=1, alpha=0.8)

# Orbit plot
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

earth_dot, = ax.plot([], [], 'o', color='#4aa3ff', markersize=6)
sun_dot, = ax.plot([], [], 'o', color='yellow', markersize=12)

trail, = ax.plot([], [], color='#00ffff', linewidth=2, alpha=0.7)
full_path, = ax.plot([], [], color='#1f77b4', alpha=0.2)

# HUD text
coord_text = ax.text(
    0.02, 0.90,
    '',
    transform=ax.transAxes,
    fontsize=10,
    color='white',
    family='monospace'
)

ax.set_title("2-Body Simulation", color='white')
ax.set_xlabel("X (AU)", color='white')
ax.set_ylabel("Y (AU)", color='white')
ax.tick_params(colors='white')
ax.set_aspect('equal')

# Energy plot
ax_energy.set_title("Energy Conservation", color='white')
ax_energy.set_xlabel("Time Step", color='white')
ax_energy.set_ylabel("Energy (J)", color='white')
ax_energy.tick_params(colors='white')

ke_line, = ax_energy.plot([], [], color='orange', label="KE")
pe_line, = ax_energy.plot([], [], color='red', label="PE")
te_line, = ax_energy.plot([], [], color='white', label="Total")

ax_energy.legend()


# Animation Function
def update(frame):

    ex, ey = earth_positions[frame]
    sx, sy = sun_positions[frame]

    earth_dot.set_data([ex], [ey])
    sun_dot.set_data([sx], [sy])

    start = max(0, frame - 150)

    trail.set_data(
        earth_positions[start:frame, 0],
        earth_positions[start:frame, 1]
    )

    full_path.set_data(
        earth_positions[:frame, 0],
        earth_positions[:frame, 1]
    )

    speed = earth_speeds[frame]

    coord_text.set_text(
        f"EARTH (AU): ({ex:.3f}, {ey:.3f})\n"
        f"SUN   (AU): ({sx:.6f}, {sy:.6f})\n"
        f"SPEED: {speed:.0f} m/s"
    )

    # Energy
    ke_line.set_data(range(frame), energies[:frame, 0])
    pe_line.set_data(range(frame), energies[:frame, 1])
    te_line.set_data(range(frame), energies[:frame, 2])

    ax_energy.relim()
    ax_energy.autoscale_view()

    return earth_dot, sun_dot, trail, full_path, coord_text, ke_line, pe_line, te_line


# Animation
ani = FuncAnimation(fig, update, frames=len(earth_positions), interval=15, blit=False)

plt.tight_layout()
plt.show()