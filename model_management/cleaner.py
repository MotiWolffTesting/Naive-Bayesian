class Cleaner:
    """Provides data cleaning utilities, including Laplace smoothing parameters."""
    def __init__(self, laplace_alpha: float = 1.0):
        self.laplace_alpha = laplace_alpha

    def get_laplace_alpha(self) -> float:
        return self.laplace_alpha
