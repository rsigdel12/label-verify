class InvalidImageError(ValueError):
    """The uploaded bytes are not a supported, decodable image."""


class ExtractionUnavailableError(RuntimeError):
    """No configured extraction backend could process the image."""
