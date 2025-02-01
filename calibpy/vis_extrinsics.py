import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_extrinsics(calibration_file_path):
    """
    Visualizes the spatial relationship (extrinsics) between a camera and a projector
    based on the stereo calibration results in a JSON file.

    Args:
        calibration_file_path (str): Path to the JSON file containing calibration results.
    """
    with open(calibration_file_path, "r") as f:
        calibration_data = json.load(f)

    # Extract extrinsics
    camera_Rt = np.array(calibration_data["camera"]["Rt"])
    projector_Rt = np.array(calibration_data["projector"]["Rt"])

    # Camera pose is considered as the origin (world frame)
    camera_R = np.eye(3)
    camera_T = np.zeros((3, 1))

    # Projector pose is given relative to the camera
    # Assuming projector_Rt is the transformation from camera frame to projector frame
    projector_R = projector_Rt[:3, :3]
    projector_T = projector_Rt[:3, 3].reshape(3, 1)

    # Convert mm to cm for better visualization
    camera_T = camera_T / 10.0  # convert to cm
    projector_T = projector_T / 10.0  # convert to cm

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Function to draw a pyramid representing a camera
    def draw_pyramid(ax, R, T, color, label):
        # Define pyramid vertices in camera coordinate system (in cm)
        scale = 3.0  # Reduced scale for better fit
        vertices = np.array(
            [
                [0, 0, 0],  # apex
                [-scale, -scale, 2*scale],  # base 1
                [scale, -scale, 2*scale],  # base 2
                [scale, scale, 2*scale],  # base 3
                [-scale, scale, 2*scale],  # base 4
            ]
        )

        # Apply rotation and translation
        transformed_vertices = (R @ vertices.T + T).T
        
        # Store vertices for plotting bounds calculation
        draw_pyramid.all_points.append(transformed_vertices)

        # Define pyramid edges
        edges = [
            [vertices[0], vertices[1]],
            [vertices[0], vertices[2]],
            [vertices[0], vertices[3]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[4]],
            [vertices[4], vertices[1]],
        ]

        transformed_edges = [
            [
                (R @ edge[0].reshape(3, 1) + T).flatten(),
                (R @ edge[1].reshape(3, 1) + T).flatten(),
            ]
            for edge in edges
        ]

        # Plot the pyramid
        for edge in transformed_edges:
            p1 = edge[0]
            p2 = edge[1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)

        # Label the camera
        apex_transformed = (R @ vertices[0].reshape(3, 1) + T).flatten()
        ax.text(
            apex_transformed[0],
            apex_transformed[1],
            apex_transformed[2],
            label,
            color=color,
        )

    # Initialize storage for all vertices
    draw_pyramid.all_points = []

    # Draw camera pyramid (at origin)
    draw_pyramid(ax, camera_R, camera_T, "blue", "Camera")

    # Draw projector pyramid (transformed)
    draw_pyramid(ax, projector_R, projector_T, "red", "Projector")

    # Calculate bounds using all points including pyramid vertices
    all_points = np.vstack(draw_pyramid.all_points)
    max_vals = np.max(all_points, axis=0)
    min_vals = np.min(all_points, axis=0)
    
    # Set axis limits with some padding
    padding = 2.0  # cm
    ax.set_xlim(min_vals[0] - padding, max_vals[0] + padding)
    ax.set_ylim(min_vals[1] - padding, max_vals[1] + padding)
    ax.set_zlim(min_vals[2] - padding, max_vals[2] + padding)

    # Set axis labels and title
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title("Camera-Projector Extrinsics (cm)")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    calibration_file = "calibpy\\data\\results\\stereo_calib_result.json"
    visualize_extrinsics(calibration_file)
