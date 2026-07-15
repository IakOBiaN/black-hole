"""Pinhole camera that looks at the black hole from a given distance."""

import numpy as np


class Camera:
    def __init__(self, distance, resolution=(320, 320), fov_deg=28.0,
                 inclination_deg=0.0, world_up=(0.0, 0.0, 1.0),
                 aim_deg=(0.0, 0.0), roll_deg=0.0):
        """aim_deg offsets the view center from the hole (right, up) in
        degrees, for close-up framing; roll_deg tilts the camera about its
        line of sight (positive rolls the scene counterclockwise)."""
        inc = np.radians(inclination_deg)
        self.position = distance * np.array([np.cos(inc), 0.0, np.sin(inc)])
        self.width, self.height = resolution
        self.fov = np.radians(fov_deg)
        self.aim = tuple(np.tan(np.radians(a)) for a in aim_deg)
        self._params = (distance, fov_deg, inclination_deg, world_up,
                        aim_deg, roll_deg)

        forward = -self.position / np.linalg.norm(self.position)
        world_up = np.asarray(world_up, dtype=float)
        right = np.cross(forward, world_up)
        norm = np.linalg.norm(right)
        if norm < 1.0e-9:
            # Looking along world_up (pole-on view): fall back to +x as up.
            right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
            norm = np.linalg.norm(right)
        right /= norm
        up = np.cross(right, forward)

        roll = np.radians(roll_deg)
        if roll != 0.0:
            c, s = np.cos(roll), np.sin(roll)
            right, up = c * right - s * up, s * right + c * up

        self.forward, self.right, self.up = forward, right, up

    def supersampled(self, factor):
        """A copy of this camera with resolution scaled up by an integer
        factor, for supersampled anti-aliasing."""
        distance, fov_deg, inclination_deg, world_up, aim, roll = self._params
        return Camera(distance, (self.width * factor, self.height * factor),
                      fov_deg, inclination_deg, world_up, aim, roll)

    def ray_directions(self):
        """Unit ray direction for every pixel, shape (height, width, 3)."""
        half_h = np.tan(self.fov / 2.0)
        half_w = half_h * self.width / self.height

        xs = (np.arange(self.width) + 0.5) / self.width * 2.0 - 1.0
        ys = 1.0 - (np.arange(self.height) + 0.5) / self.height * 2.0
        gx, gy = np.meshgrid(xs * half_w + self.aim[0],
                             ys * half_h + self.aim[1])

        dirs = (self.forward[None, None, :]
                + gx[..., None] * self.right[None, None, :]
                + gy[..., None] * self.up[None, None, :])
        dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
        return dirs
