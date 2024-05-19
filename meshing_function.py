import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def generate_mesh(surface_points):
    """
    Generate a triangular mesh for a non-regular surface.

    Parameters:
        surface_points (array-like): List or array of surface points (x, y, z).

    Returns:
        tri (array-like): Triangular mesh indices.
    """
    # Perform Delaunay triangulation
    tri = Delaunay(surface_points[:, :2])

    print("TRI")
    print(type(tri))
    print(tri)

    return tri


def plot_mesh(surface_points, tri):
    """
    Plot the meshed surface.

    Parameters:
        surface_points (array-like): List or array of surface points (x, y, z).
        tri (array-like): Triangular mesh indices.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface points
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], c='r', marker='o')

    # Plot mesh triangles
    ax.plot_trisurf(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], triangles=tri.simplices,
                    edgecolor='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


# Example usage
# Generate some random non-regular surface points
surface_points = np.random.rand(30, 3)  # 30 points with (x, y, z) coordinates
# print("surface_points", surface_points)
# print(type(surface_points))
# Generate mesh
tri = generate_mesh(surface_points)
# Plot mesh
plot_mesh(surface_points, tri)

