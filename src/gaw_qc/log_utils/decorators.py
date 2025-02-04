import logging
import uuid
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def log_function(
    logger: logging.Logger, show_args: bool = False, show_return: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def log_function_wrapper(func: Callable[P, T]) -> Callable[P, T]:
        """
        Decorator to log the calling of a function
        """

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            fun_name = func.__name__
            execution_id = uuid.uuid4()
            extra = f"with {args} and {kwargs}" if show_args else ""
            logger.debug(f"{execution_id}: Executing {fun_name}{extra}.")
            res = func(*args, **kwargs)
            ret = f"with return {res}" if show_return else ""
            logger.debug(
                f"{execution_id}: Finished function execution of {fun_name}{ret}."
            )
            return res

        return wrapper

    return log_function_wrapper
