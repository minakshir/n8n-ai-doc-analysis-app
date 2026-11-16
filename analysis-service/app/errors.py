class ServiceError(Exception):
    """An expected failure that is reported to the caller as {"error": {...}}."""

    def __init__(self, code: str, message: str, hint: str | None = None, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.status = status

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "hint": self.hint}}
