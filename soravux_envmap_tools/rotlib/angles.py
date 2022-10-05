import numpy as np


# Taken from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def angle_between_vectors(v0, v1, directed=True, axis=0):
    """Return angle between vectors.

    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.

    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    >>> numpy.allclose(a, math.pi)
    True
    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3], directed=False)
    >>> numpy.allclose(a, 0)
    True
    >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    >>> v1 = [[3], [0], [0]]
    >>> a = angle_between_vectors(v0, v1)
    >>> numpy.allclose(a, [0, 1.5708, 1.5708, 0.95532])
    True
    >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    >>> a = angle_between_vectors(v0, v1, axis=1)
    >>> numpy.allclose(a, [1.5708, 1.5708, 1.5708, 0.95532])
    True

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)
    dot = numpy.sum(v0 * v1, axis=axis)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    return numpy.arccos(dot if directed else numpy.fabs(dot))


def angularDifferenceSphZenith(theta1, phi1, theta2, phi2):
    """Gets the angular distance from spherical coordinates, where `theta` is the zenith angle."""
    return np.arccos(np.cos(theta1)*np.cos(theta2) + np.sin(theta1)*np.sin(theta2)*np.cos(phi1 - phi2))


def angularDifferenceSphElevation(theta1, phi1, theta2, phi2):
    """Gets the angular distance from spherical coordinates, where `theta` is the elevation angle."""
    return angularDifferenceSphZenith(np.pi/2 - theta1, phi1, np.pi/2 - theta2, phi2)


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def sph2cart(azimuth, elevation, r):
    x = r*np.cos(elevation)*np.cos(azimuth)
    y = r*np.cos(elevation)*np.sin(azimuth)
    z = r*np.sin(elevation)
    return x, y, z
