"""Outcome of model registration and sample settlement for one update."""

MAX_ATTEMPTS = 3
RETRY_DELAY_SEC = 0.05


class UpdateCommitError(RuntimeError):
    def __init__(self, message: str, attempts: int = 0):
        super().__init__(message)
        self.attempts = int(attempts)


class UpdateCommitOutcomeUncertain(UpdateCommitError):
    """The remote side effect may have succeeded despite the RPC failure."""


class UpdateCommitNotApplied(UpdateCommitError):
    """The exact remote state proves that the side effect was not applied."""
