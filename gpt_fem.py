import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, MatrixRankWarning
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings


def generate_half_sphere_mesh(radius, num_points):
    # Generate points on a half-sphere
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    points = points[z.ravel() >= 0]  # Only keep points on the upper hemisphere

    # Create a 2D projection for Delaunay triangulation
    points_2d = points[:, :2]
    tri = Delaunay(points_2d)

    return points, tri.simplices


def assemble_system(points, simplices):
    num_points = len(points)

    # Initialize arrays to store matrix entries
    data = []
    row_indices = []
    col_indices = []
    rhs = np.zeros(num_points)

    # Loop over each element (triangle)
    for simplex in simplices:
        pts = points[simplex]

        # Compute the area of the triangle
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if area == 0:
            print("area = 0", area)
            continue  # Skip degenerate triangles

        # Compute local stiffness matrix for the triangle
        B = np.array([
            [pts[1, 0] - pts[0, 0], pts[2, 0] - pts[0, 0]],
            [pts[1, 1] - pts[0, 1], pts[2, 1] - pts[0, 1]]
        ])
        # C = np.linalg.inv(B) @ np.array([[-1, -1], [1, 0], [0, 1]])
        # local_stiffness = (area / 2) * (C @ C.T)

        try:
            invB = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print(f"Skipping degenerate triangle with points {pts}")
            continue

        grads = np.array([[-1, -1], [1, 0], [0, 1]])
        C = invB @ grads.T  # (2x2) @ (2x3) -> (2x3)

        local_stiffness = (area / 2) * (C.T @ C)  # (3x2) @ (2x3) -> (3x3)

        # Assemble the global stiffness matrix
        for i in range(3):
            for j in range(3):
                row_indices.append(simplex[i])
                col_indices.append(simplex[j])
                data.append(local_stiffness[i, j])

        # Assemble the global load vector (rhs)
        for i in range(3):
            rhs[simplex[i]] += area / 3.0  # Assuming a unit load for simplicity

    # Create the sparse global stiffness matrix
    stiffness_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_points, num_points))

    # Debugging: print the non-zero elements of the stiffness matrix
    print("Stiffness Matrix (non-zero elements):")
    for i, j, v in zip(row_indices, col_indices, data):
        print(f"K[{i}, {j}] = {v}")

    return stiffness_matrix, rhs


# def apply_boundary_conditions(stiffness_matrix, rhs, boundary_nodes):
#     # Apply Dirichlet boundary conditions by modifying the system matrix and RHS vector
#     for node in boundary_nodes:
#         for j in range(stiffness_matrix.shape[1]):
#             if node != j:
#                 rhs[j] -= stiffness_matrix[j, node] * 0.0
#         stiffness_matrix[node, :] = 0
#         stiffness_matrix[:, node] = 0
#         stiffness_matrix[node, node] = 1
#         rhs[node] = 0.0
#     return stiffness_matrix, rhs

def apply_boundary_conditions(stiffness_matrix, rhs, boundary_nodes):
    for node in boundary_nodes:
        stiffness_matrix[node, :] = 0
        stiffness_matrix[:, node] = 0
        stiffness_matrix[node, node] = 1
        rhs[node] = 0.0
    return stiffness_matrix, rhs


def solve_fem(stiffness_matrix, rhs):
    # Solve the linear system
    # solution = spsolve(stiffness_matrix.tocsr(), rhs)
    # return solution
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        try:
            solution = spsolve(stiffness_matrix.tocsr(), rhs)
        except MatrixRankWarning:
            print("Warning: Matrix is singular, unable to solve.")
            solution = None
    return solution


def check_connectivity(points, simplices):
    num_points = len(points)
    connected = np.zeros(num_points, dtype=bool)

    for simplex in simplices:
        connected[simplex] = True

    isolated_nodes = np.where(~connected)[0]
    if len(isolated_nodes) > 0:
        print(f"Isolated nodes detected: {isolated_nodes}")
    else:
        print("All nodes are connected.")


def visualize_stiffness_matrix(stiffness_matrix):
    plt.spy(stiffness_matrix, markersize=1)
    plt.title("Stiffness Matrix Structure")
    plt.show()


def remove_isolated_nodes(points, simplices):
    """
    Remove isolated nodes by ensuring every node is part of at least one simplex.
    """
    num_points = len(points)
    connected = np.zeros(num_points, dtype=bool)

    for simplex in simplices:
        connected[simplex] = True

    isolated_nodes = np.where(~connected)[0]
    if len(isolated_nodes) > 0:
        print(f"Isolated nodes detected and removed: {isolated_nodes}")
        mask = np.ones(num_points, dtype=bool)
        mask[isolated_nodes] = False
        points = points[mask]

        # Update simplices to reflect removal of isolated nodes
        remap = np.cumsum(mask) - 1
        simplices = remap[simplices]

    return points, simplices


def main():
    radius = 1.0
    num_points = 30
    points, simplices = generate_half_sphere_mesh(radius, num_points)

    # Visualize the mesh for debugging
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=simplices, cmap='viridis', alpha=0.5)
    plt.title("Generated Mesh")
    plt.show()

    # Check node connectivity
    check_connectivity(points, simplices)

    # Remove isolated nodes
    points, simplices = remove_isolated_nodes(points, simplices)

    # Check node connectivity again
    check_connectivity(points, simplices)

    # For simplicity, assume boundary nodes are those on the edge of the half-sphere
    boundary_nodes = np.where(np.isclose(points[:, 2], 0))[0]

    stiffness_matrix, rhs = assemble_system(points, simplices)
    # stiffness_matrix, rhs = apply_boundary_conditions(stiffness_matrix, rhs, boundary_nodes)

    # Visualize the stiffness matrix structure
    visualize_stiffness_matrix(stiffness_matrix)

    # Check for singular matrix before applying boundary conditions
    if np.linalg.matrix_rank(stiffness_matrix.todense()) < stiffness_matrix.shape[0]:
        print("Warning: Stiffness matrix is singular before applying boundary conditions.",
              np.linalg.matrix_rank(stiffness_matrix.todense()), " < ", stiffness_matrix.shape[0])

    stiffness_matrix, rhs = apply_boundary_conditions(stiffness_matrix.tolil(), rhs, boundary_nodes)

    # Check for singular matrix after applying boundary conditions
    if np.linalg.matrix_rank(stiffness_matrix.todense()) < stiffness_matrix.shape[0]:
        print("Warning: Stiffness matrix is singular after applying boundary conditions.",
              np.linalg.matrix_rank(stiffness_matrix.todense()), " < ", stiffness_matrix.shape[0])

    solution = solve_fem(stiffness_matrix, rhs)

    return points, simplices, solution


points, simplices, solution = main()

# Plotting the solution
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=simplices, cmap='viridis', alpha=0.5)
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=solution, cmap='viridis')
# plt.show()

if solution is not None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=simplices, cmap='viridis', alpha=0.5)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=solution, cmap='viridis', edgecolor='k')
    plt.title("FEM Solution")
    plt.show()
else:
    print("Failed to solve the FEM problem due to singular matrix.")