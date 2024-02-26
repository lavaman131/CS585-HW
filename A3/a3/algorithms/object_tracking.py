from typing import Tuple
import numpy as np
from a3 import MISSING_VALUE
import cv2
import cv2.typing


class AlphaBetaFilter2D:
    def __init__(
        self,
        alpha: float,
        beta: float,
        x_0: float = 0.0,
        y_0: float = 0.0,
        v_x_0: float = 0.0,
        v_y_0: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        """Initializes the alpha-beta filter for 2D coordinates.

        Parameters
        ----------
        alpha : float
            Alpha parameter for the alpha-beta filter. A higher alpha value is more sensitive to changes in the measurements.
        beta : float, optional
            Beta parameter for the alpha-beta filter. A higher beta value is more sensitive to changes in the measurements.
        x_0 : float, optional
            Initial x position of the object, by default 0.0
        y_0 : float, optional
            Initial y position of the object, by default 0.0
        v_x_0 : float, optional
            Initial x velocity of the object, by default 0.0
        v_y_0 : float, optional
            Initial y velocity of the object, by default 0.0
        dt : float, optional
            Time step between frames (seconds), by default 1.0
        """
        self.alpha = alpha
        self.beta = beta
        self.x_0 = x_0
        self.y_0 = y_0
        self.v_x_0 = v_x_0
        self.v_y_0 = v_y_0
        self.dt = dt

        # Initialize the state variables
        self.x_k = self.x_0
        self.v_x_k = self.v_x_0
        self.y_k = self.y_0
        self.v_y_k = self.v_y_0

    def __call__(self, measurement: Tuple[int, int]) -> Tuple[int, int]:
        """Returns the corrected 2D coordinates of the object.

        Parameters
        ----------
        measurement : Tuple[int, int]
            2D coordinates of the object.

        Returns
        -------
        corrected_measurement : Tuple[int, int]
            Corrected 2D coordinates of the object.
        """
        z_x_k, z_y_k = measurement
        # Calculate the predicted state
        self.x_k = self.x_k + self.dt * self.v_x_k
        self.v_x_k = self.v_x_k

        self.y_k = self.y_k + self.dt * self.v_y_k
        self.v_y_k = self.v_y_k

        # If the measurement is not missing
        if z_x_k != MISSING_VALUE:
            # Calculate the error
            e_k = z_x_k - self.x_k

            # Update the state
            self.x_k += self.alpha * e_k
            self.v_x_k += (self.beta * e_k) / self.dt

        if z_y_k != MISSING_VALUE:
            # Calculate the error
            e_k = z_y_k - self.y_k

            # Update the state
            self.y_k += self.alpha * e_k
            self.v_y_k += (self.beta * e_k) / self.dt

        corrected_measurement = (int(self.x_k), int(self.y_k))

        return corrected_measurement

    def predict(self, measurements: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Predicts the 2D coordinates of the object.

        Parameters
        ----------
        measurements : cv2.typing.MatLike
            Array of 2D coordinates of the object.

        Returns
        -------
        predicted_measurements : cv2.typing.MatLike
            Predicted 2D coordinates of the object.
        """
        predicted_measurements = np.zeros_like(measurements)

        for i, measurement in enumerate(measurements):
            predicted_measurements[i] = self(measurement)

        return predicted_measurements
