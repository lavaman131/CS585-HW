from typing import List
from ..algorithms import MISSING_VALUE


def alpha_beta_filter(
    coords: List[int],
    alpha: float,
    beta: float,
    x_0: float = 0.0,
    v_0: float = 0.0,
    dt: float = 1.0,
) -> List[int]:
    """Cleans and interpolates the object coordinates using alpha-beta filter as described in https://en.wikipedia.org/wiki/Alpha_beta_filter.

    Parameters
    ----------
    coords : List[int]
        List of coordinates of object.
    alpha : float
        Alpha parameter for the alpha-beta filter. A higher alpha value is more sensitive to changes in the measurements.
    beta : float, optional
        Beta parameter for the alpha-beta filter. A higher beta value is more sensitive to changes in the measurements.
    x_0 : float, optional
        Initial position of the object, by default 0.0
    v_0 : float, optional
        Initial velocity of the object, by default 0.0
    dt : float, optional
        Time step between frames (seconds), by default 1.0

    Returns
    -------
    corrected_coords : List[int]
        List of cleaned and interpolated coordinates of the object.
    """

    # Initialize the state variables
    x_k = x_0
    v_k = v_0

    # Initialize the corrected coordinates
    corrected_coords = []

    n = len(coords)

    for i in range(n):
        # Get the current measurement
        z_k = coords[i]

        # Calculate the predicted state
        x_k = x_k + dt * v_k
        v_k = v_k

        # If the measurement is not missing
        if z_k != MISSING_VALUE:
            # Calculate the error
            e_k = z_k - x_k

            # Update the state
            x_k = x_k + alpha * e_k
            v_k = v_k + (beta * e_k) / dt

        # Append the corrected object coordinate
        corrected_coords.append(int(x_k))

    return corrected_coords


def alpha_beta_filter_2d(
    coords: List[List[int]],
    alpha: float,
    beta: float,
    x_0: float = 0.0,
    v_0: float = 0.0,
    dt: float = 1.0,
) -> List[List[int]]:
    """Cleans and interpolates the object coordinates using alpha-beta filter for 2D coordinates.

    Parameters
    ----------
    coords : List[List[int]]
        List of 2D coordinates of object.
    alpha : float
        Alpha parameter for the alpha-beta filter. A higher alpha value is more sensitive to changes in the measurements.
    alpha : float
        Alpha parameter for the alpha-beta filter. A higher alpha value is more sensitive to changes in the measurements.
    beta : float, optional
        Beta parameter for the alpha-beta filter. A higher beta value is more sensitive to changes in the measurements.
    x_0 : float, optional
        Initial position of the object, by default 0.0
    v_0 : float, optional
        Initial velocity of the object, by default 0.0
    dt : float, optional
        Time step between frames (seconds), by default 1.0

    Returns
    -------
    corrected_coords : List[int]
        List of cleaned and interpolated 2D coordinates of the object.
    """
    x_coords, y_coords = zip(*coords)
    corrected_x_coords = alpha_beta_filter(x_coords, alpha, beta, x_0, v_0, dt)
    corrected_y_coords = alpha_beta_filter(y_coords, alpha, beta, x_0, v_0, dt)

    corrected_coords = []
    for i in range(len(corrected_x_coords)):
        corrected_coords.append([corrected_x_coords[i], corrected_y_coords[i]])

    return corrected_coords
