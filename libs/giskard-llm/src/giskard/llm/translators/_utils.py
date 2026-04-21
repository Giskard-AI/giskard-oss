def require[T](value: T | None, message: str) -> T:
    if value is None:
        raise ValueError(message)
    return value
