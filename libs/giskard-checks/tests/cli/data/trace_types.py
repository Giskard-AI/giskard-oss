from giskard.checks import Trace


class CustomTrace(Trace[str, str], frozen=True):
    pass
