from function import *
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_hand_landmarks(action, sequence, frame_num):
    # Load the keypoints from the .npy file
    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + '.npy')
    keypoints = np.load(npy_path)

    # Extract 3D coordinates from the loaded keypoints for both hands
    rh_3d = keypoints[:63].reshape((21, 3))  # Right hand

    # Check if landmarks for the left hand exist before attempting to reshape
    if len(keypoints) > 63:
        lh_3d = keypoints[63:].reshape((21, 3))  # Left hand
    else:
        lh_3d = None  # No left-hand landmarks

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot right-hand landmarks
    ax.scatter(rh_3d[:, 0], rh_3d[:, 1], rh_3d[:, 2], marker='o', s=50, c='r', label='Right Hand Landmarks')

    # Plot connections between landmarks for the right hand
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        ax.plot([rh_3d[start_point, 0], rh_3d[end_point, 0]],
                [rh_3d[start_point, 1], rh_3d[end_point, 1]],
                [rh_3d[start_point, 2], rh_3d[end_point, 2]], c='b')

    # Plot left-hand landmarks and connections if they exist
    if lh_3d is not None:
        ax.scatter(lh_3d[:, 0], lh_3d[:, 1], lh_3d[:, 2], marker='o', s=50, c='g', label='Left Hand Landmarks')
        for connection in connections:
            start_point = connection[0] + 21
            end_point = connection[1] + 21
            ax.plot([lh_3d[start_point-21, 0], lh_3d[end_point-21, 0]],
                    [lh_3d[start_point-21, 1], lh_3d[end_point-21, 1]],
                    [lh_3d[start_point-21, 2], lh_3d[end_point-21, 2]], c='b')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot title
    ax.set_title(f'3D Hand Landmarks with Connections - Action: {action}, Sequence: {sequence}, Frame: {frame_num}')

    # Display the plot
    plt.show()

# Example usage
action = 'Absen'
sequence = 9
frame_num = 0

visualize_hand_landmarks(action, sequence, frame_num)
