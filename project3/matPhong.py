class MatPhong:
    def __init__(self, ka: float, kd: float, ks: float, n: float) -> None:
        """
        Initialize a Phong material model.

        Parameters:
        - ka (float): Ambient reflection coefficient.
        - kd (float): Diffuse reflection coefficient.
        - ks (float): Specular reflection coefficient.
        - n  (float): Phong exponent (shininess).
        """
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.n = n
