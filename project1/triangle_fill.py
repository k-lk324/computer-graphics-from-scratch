import numpy as np


def vector_interp(p1, p2, V1, V2, coord, dim):
    """
    Linearly interpolates a vector value along a segment between p1 and p2.

    Parameters:
        p1, p2: np.array of shape (2,) - points defining the segment.
        V1, V2: np.array - values at p1 and p2.
        coord: float - the x or y coordinate to interpolate at.
        dim: int - 1 for x interpolation, 2 for y.

    Returns:
        Interpolated vector (same shape as V1).
    """
    denominator = p2[dim - 1] - p1[dim - 1]
    if denominator == 0:
        return V1  # Avoid division by zero, return V1 if p1 and p2 are the same
    # Calculate the interpolation factor
    t = (coord - p1[dim - 1]) / denominator

    # Interpolate the vector value
    return V1 + t * (V2 - V1)


def edge_function(v0, v1, p):
    """
    Computes the edge function for a given edge and points.

    Args:
        v0 (array-like): Start vertex of the edge (x, y).
        v1 (array-like): End vertex of the edge (x, y).
        p (np.ndarray): Points to evaluate, shape (N, 2).

    Returns:
        np.ndarray: Signed areas, positive if point is on the left of the edge.
    """
    edge_vec = v1 - v0
    pt_vec = p - v0
    return edge_vec[0] * pt_vec[..., 1] - edge_vec[1] * pt_vec[..., 0]


def f_shading(img, vertices, vcolors):
    """
    Flat shading - fill triangle with the average of vertex colors.

    Parameters:
        img: np.array (H, W, 3) - image buffer.
        vertices: np.array (3, 2) - triangle vertex coordinates.
        vcolors: np.array (3, 3) - RGB colors of the vertices.

    Returns:
        Updated image with filled triangle.
    """
    # Compute the bounding box of the triangle
    min_x = max(0, int(np.floor(np.min(vertices[:, 0]))))
    max_x = min(img.shape[1], int(np.ceil(np.max(vertices[:, 0]))))
    min_y = max(0, int(np.floor(np.min(vertices[:, 1]))))
    max_y = min(img.shape[0], int(np.ceil(np.max(vertices[:, 1]))))

    # Compute the average color
    avg_color = np.mean(vcolors, axis=0)

    # Create a grid of pixel coordinates within the bounding box
    x, y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
    coords = np.stack((x.ravel(), y.ravel()), axis=-1)

    area = edge_function(vertices[0], vertices[1], vertices[2][None, :])
    if area != 0:
        w0 = edge_function(vertices[1], vertices[2], coords) / area
        w1 = edge_function(vertices[2], vertices[0], coords) / area
        w2 = edge_function(vertices[0], vertices[1], coords) / area

        # Mask for pixels inside the triangle
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

        # Update the image with the average color for pixels inside the triangle
        img[coords[mask, 1], coords[mask, 0]] = 255 * avg_color

    return img


def t_shading(img, vertices, uv, textImg):
    """
    Texture shading - interpolate UVs and sample texture.

    Parameters:
        img: np.array (H, W, 3) - image buffer.
        vertices: np.array (3, 2) - triangle vertex coordinates.
        uv: np.array (3, 2) - UV texture coordinates.
        textImg: np.array (K, L, 3) - texture image.

    Returns:
        Updated image with textured triangle.
    """

    updated_img = img.copy()
    height, width, _ = img.shape
    tex_height, tex_width, _ = textImg.shape

    # Sort vertices by y-coordinate (ascending)
    sorted_indices = np.argsort(vertices[:, 1])
    vertices = vertices[sorted_indices]
    uv = uv[sorted_indices]

    y_min = int(np.ceil(vertices[0][1]))
    y_max = int(np.floor(vertices[2][1]))

    for y in range(y_min, y_max + 1):
        # Skip scanlines outside the image
        if y < 0 or y >= height:
            continue

        # Interpolate along the edges of the triangle
        if vertices[1][1] <= y:  # bottom half
            A = vector_interp(
                vertices[0], vertices[2], vertices[0], vertices[2], y, dim=2
            )
            uv_A = vector_interp(
                vertices[0], vertices[2], uv[0], uv[2], y, dim=2
            )

            B = vector_interp(
                vertices[1], vertices[2], vertices[1], vertices[2], y, dim=2
            )
            uv_B = vector_interp(
                vertices[1], vertices[2], uv[1], uv[2], y, dim=2
            )
        else:  # top half
            A = vector_interp(
                vertices[0], vertices[2], vertices[0], vertices[2], y, dim=2
            )
            uv_A = vector_interp(
                vertices[0], vertices[2], uv[0], uv[2], y, dim=2
            )

            B = vector_interp(
                vertices[0], vertices[1], vertices[0], vertices[1], y, dim=2
            )
            uv_B = vector_interp(
                vertices[0], vertices[1], uv[0], uv[1], y, dim=2
            )

        # Ensure A is left of B
        if A[0] > B[0]:
            A, B = B, A
            uv_A, uv_B = uv_B, uv_A

        x_min = int(np.ceil(A[0]))
        x_max = int(np.floor(B[0]))

        for x in range(x_min, x_max + 1):
            if x < 0 or x >= width:
                continue

            # Interpolate uv at (x, y)
            uv_P = vector_interp(A, B, uv_A, uv_B, x, dim=1)

            # Map uv to texture coordinates
            tex_x = int(round(uv_P[0] * (tex_width - 1)))
            tex_y = int(round((1 - uv_P[1]) * (tex_height - 1)))

            # Clamp texture coordinates
            tex_x = np.clip(tex_x, 0, tex_width - 1)
            tex_y = np.clip(tex_y, 0, tex_height - 1)

            # Apply texture color
            updated_img[y, x] = textImg[tex_y, tex_x]

    return updated_img


def render_img(faces, vertices, vcolors, uvs, depth, shading, textImg=None):
    """
    Renders a 3D object as a 2D image based on triangle definitions.

    Parameters:
        faces: np.array (K, 3) - triangle vertex indices.
        vertices: np.array (L, 2) - 2D coordinates of all vertices.
        vcolors: np.array (L, 3) - RGB vertex colors.
        uvs: np.array (L, 2) - UV coordinates.
        depth: np.array (L,) - depth values for each vertex.
        shading: str - 'f' for flat, 't' for texture.
        textImg: np.array (K, L, 3) - optional texture image.

    Returns:
        Rendered image of size (512, 512, 3).
    """
    img = np.ones((512, 512, 3), dtype=np.float32) * 255  # white background

    # Compute depth of each triangle and sort
    triangle_depths = np.mean(depth[faces], axis=1)
    sorted_indices = np.argsort(-triangle_depths)  # farthest first

    for idx in sorted_indices:
        face = faces[idx]
        verts = vertices[face]
        colors = vcolors[face]

        if shading == "f":
            img = f_shading(img, verts, colors)
        elif shading == "t" and textImg is not None:
            uv_coords = uvs[face]
            img = t_shading(img, verts, uv_coords, textImg)

    return img
