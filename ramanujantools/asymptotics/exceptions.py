class PrecisionExhaustedError(Exception):
    """Base class for all precision-related asymptotic engine bounds."""

    def __init__(self, required_precision: int, message: str):
        self.required_precision = required_precision
        super().__init__(
            f"{message} [REQUIRED_STARTING_PRECISION: {required_precision}]"
        )


class EigenvalueBlindnessError(PrecisionExhaustedError):
    """Raised when the matrix appears nilpotent at the current precision."""

    pass


class RowNullityError(PrecisionExhaustedError):
    """Raised when a physical variable completely vanishes from the formal solution space."""

    pass


class InputTruncationError(PrecisionExhaustedError):
    """Raised when the starting precision is too low to fully ingest the input matrix."""

    pass
