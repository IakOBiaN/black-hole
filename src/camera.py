"""Pinhole camera that looks at the black hole from a given distance."""

import numpy as np


class Camera:
    def __init__(self, distance, resolution=(320, 320), fov_deg=28.0,
                 inclination_deg=0.0, world_up=(0.0, 0.0, 1.0)):
        inc = np.radians(inclination_deg)
        self.position = distance * np.array([np.cos(inc), 0.0, np.sin(inc)])
        self.width, self.height = resolution
        self.fov = np.radians(fov_deg)

        forward = -self.position / np.linalg.norm(self.position)
        world_up = np.asarray(world_up, dtype=float)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        self.forward, self.right, self.up = forward, right, up

    def ray_directions(self):
        """Unit ray direction for every pixel, shape (height, width, 3)."""
        half_h = np.tan(self.fov / 2.0)
        half_w = half_h * self.width / self.height

        xs = (np.arange(self.width) + 0.5) / self.width * 2.0 - 1.0
        ys = 1.0 - (np.arange(self.height) + 0.5) / self.height * 2.0
        gx, gy = np.meshgrid(xs * half_w, ys * half_h)

        dirs = (self.forward[None, None, :]
                + gx[..., None] * self.right[None, None, :]
                + gy[..., None] * self.up[None, None, :])
        dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
        return dirs
