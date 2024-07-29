import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle

# Load the data
data = pd.read_csv('data1.csv')

# Define room positions based on the floor plan
room_positions = {
    'r1': (0, 5), 'r2': (0, 4), 'r3': (0, 3), 'r4': (0, 1), 'r5': (1, 1), 'r6': (0, 0),
    'r7': (2, 5), 'r8': (2, 4.5), 'r9': (2, 4), 'r10': (2, 3.5), 'r11': (2, 3), 'r12': (2, 2.5),
    'r13': (2, 2), 'r14': (2, 1), 'r15': (3, 5), 'r16': (3, 4), 'r17': (3, 3),
    'r18': (3, 2), 'r19': (4, 5), 'r20': (4, 4), 'r21': (4, 3), 'r22': (4, 1),
    'r23': (5, 5), 'r24': (5, 1), 'r25': (6, 5), 'r26': (6, 4), 'r27': (6, 2),
    'r28': (6, 1), 'r29': (7, 5), 'r30': (7, 4), 'r31': (7, 3), 'r32': (7, 2),
    'r33': (7, 1), 'r34': (7, 0),
    'c1': (5, 1.5), 'c2': (3, 3.5)  # Corridors
}

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 10))

# Initialize empty scatter plots for actual occupancy and different sensor types
scat_actual = ax.scatter([], [], s=50, c='blue', label='Actual Occupancy')
scat_motion = ax.scatter([], [], s=30, c='green', label='Motion Sensor')
scat_camera = ax.scatter([], [], s=30, c='red', label='Camera')
scat_door = ax.scatter([], [], s=30, c='purple', label='Door Sensor')

# Set axis limits
ax.set_xlim(-1, 8)
ax.set_ylim(-1, 6)

# Add room rectangles and labels
for room, pos in room_positions.items():
    rect = Rectangle((pos[0]-0.4, pos[1]-0.4), 0.8, 0.8, fill=False)
    ax.add_patch(rect)
    ax.text(pos[0], pos[1], room, ha='center', va='center')

# Add sensor indicators
motion_sensors = ['r1', 'r14', 'r19', 'r28', 'r29', 'r32']
cameras = ['r3', 'r21', 'r25', 'r34']
door_sensors = ['r2', 'r3', 'r20', 'r26', 'r28']

for room in motion_sensors:
    pos = room_positions[room]
    ax.text(pos[0]-0.4, pos[1]+0.4, '((()))', fontsize=8, color='green')

for room in cameras:
    pos = room_positions[room]
    ax.text(pos[0]+0.4, pos[1]+0.4, '🎥', fontsize=10, color='red')

for room in door_sensors:
    pos = room_positions[room]
    ax.plot([pos[0]-0.4, pos[0]+0.4], [pos[1]-0.4, pos[1]-0.4], 'purple', linestyle='--')

# Create a text object for displaying the current time
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Animation update function
def update(frame):
    positions_actual_x, positions_actual_y = [], []
    positions_motion_x, positions_motion_y = [], []
    positions_camera_x, positions_camera_y = [], []
    positions_door_x, positions_door_y = [], []
    sizes_actual, sizes_motion, sizes_camera, sizes_door = [], [], [], []
    
    for room, pos in room_positions.items():
        if room in data.columns:
            actual_count = data[room].iloc[frame]
            if actual_count > 0:
                positions_actual_x.extend([pos[0] + np.random.normal(0, 0.1) for _ in range(actual_count)])
                positions_actual_y.extend([pos[1] + np.random.normal(0, 0.1) for _ in range(actual_count)])
                sizes_actual.extend([50] * actual_count)
            
            # Handle sensor readings
            if room in motion_sensors:
                motion = data[f'motion_sensor{motion_sensors.index(room)+1}'].iloc[frame]
                if motion == 'motion':
                    positions_motion_x.append(pos[0] + 0.2)
                    positions_motion_y.append(pos[1] + 0.2)
                    sizes_motion.append(30)
            elif room in cameras:
                camera_count = data[f'camera{cameras.index(room)+1}'].iloc[frame]
                if camera_count > 0:
                    positions_camera_x.extend([pos[0] - 0.2] * camera_count)
                    positions_camera_y.extend([pos[1] - 0.2] * camera_count)
                    sizes_camera.extend([30] * camera_count)
            elif room in door_sensors:
                door_count = data[f'door_sensor{door_sensors.index(room)+1}'].iloc[frame]
                if door_count != 0:
                    positions_door_x.append(pos[0] - 0.2)
                    positions_door_y.append(pos[1] - 0.2)
                    sizes_door.append(30)
    
    scat_actual.set_offsets(np.c_[positions_actual_x, positions_actual_y])
    scat_actual.set_sizes(sizes_actual)
    scat_motion.set_offsets(np.c_[positions_motion_x, positions_motion_y])
    scat_motion.set_sizes(sizes_motion)
    scat_camera.set_offsets(np.c_[positions_camera_x, positions_camera_y])
    scat_camera.set_sizes(sizes_camera)
    scat_door.set_offsets(np.c_[positions_door_x, positions_door_y])
    scat_door.set_sizes(sizes_door)
    
    # Update the time display
    current_time = data['time'].iloc[frame]
    time_text.set_text(f"Time: {current_time}")
    
    return scat_actual, scat_motion, scat_camera, scat_door, time_text

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=len(data), interval=200, blit=True)

# Add legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

# If you want to save the animation:
# anim.save('room_occupancy_and_sensors.gif', writer='pillow', fps=5)