import logging
from types import TracebackType
from typing import Callable, Optional

logger = logging.getLogger("l2l_lab")


class ExceptionHandler:

    def __init__(self, on_exit: Callable[[], None]) -> None:
        self._on_exit = on_exit

    def __enter__(self) -> ExceptionHandler:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        if exc_type is None:
            logger.info("\nShutting down...\n")
        elif issubclass(exc_type, KeyboardInterrupt):
            logger.info("\nInterrupted by user — shutting down... (press Ctrl+C again to force quit)\n")
        else:
            logger.error("\nError raised — shutting down...\n", exc_info=(exc_type, exc, tb))

        try:
            self._on_exit()
        except Exception:
            logger.exception("Error during shutdown")

        # Swallow ordinary errors (logged above); let KeyboardInterrupt and other
        # BaseExceptions propagate so a second Ctrl+C can force-quit.
        return exc_type is not None and issubclass(exc_type, Exception)
