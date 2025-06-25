import functools
import inspect
from typing import (
    Any,
    Callable,
    Union,
    TypeVar,
    TypeAlias,
    get_origin,
    get_args,
    ForwardRef,
    _eval_type,
)
from types import UnionType

X = TypeVar("X")
Batchable: TypeAlias = X | list[X]


def batched(
    arg_name: str,
) -> Callable:
    """
    Decorator that makes a function accept either a scalar or a list for the
    specified argument name. The decorated function must accept a List[X] for
    that argument and return a List[X].

    The decorator reads the type hint of the argument and validates inputs
    accordingly, raising TypeError on mismatches.

    Parameters:
        arg_name: The name of the argument to batchify.

    Returns:
        A decorator that wraps the function accordingly.
    """

    def decorator(func: Callable) -> Callable:
        raw_annot = func.__annotations__.get(arg_name)
        if raw_annot is None:
            raise TypeError(f"No annotation for argument '{arg_name}'")

        # Now evaluate just this annotation safely
        globalns = func.__globals__
        localns = {}
        if isinstance(raw_annot, str):
            raw_annot = ForwardRef(raw_annot)
        annot = _eval_type(raw_annot, globalns, localns, recursive_guard=set())
        origin = get_origin(annot)
        args = get_args(annot)

        if origin not in (Union, UnionType) or len(args) != 2:
            raise TypeError(
                f"Parameter '{arg_name}' annotation must be X | List[X], got {annot}"
            )

        scalar_type = args[0]

        sig = inspect.signature(func)
        param = sig.parameters.get(arg_name)
        if param is None:
            raise ValueError(f"Function {func.__name__} has no parameter '{arg_name}'")

        # Assume list_type is List[scalar_type], no checks here as per your request

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Batchable:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            val = bound_args.arguments.get(arg_name)

            if val is None:
                raise ValueError(f"Argument '{arg_name}' must be provided")

            if isinstance(val, list):
                if not all(isinstance(el, scalar_type) for el in val):
                    raise TypeError(
                        f"All elements of argument '{arg_name}' must be of type Batchable[{scalar_type.__name__}] "
                        f"(i.e., {scalar_type.__name__} or list[{scalar_type.__name__}]), not {type(val).__name__}"
                    )
                return func(*args, **kwargs)
            else:
                if not isinstance(val, scalar_type):
                    raise TypeError(
                        f"Argument '{arg_name}' must be of type Batchable[{scalar_type.__name__}] "
                        f"(i.e., {scalar_type.__name__} or list[{scalar_type.__name__}]), not {type(val).__name__}"
                    )

                new_args = list(bound_args.args)
                new_kwargs = dict(bound_args.kwargs)

                param_pos = list(sig.parameters).index(arg_name)
                if param_pos < len(new_args):
                    new_args[param_pos] = [val]
                else:
                    new_kwargs[arg_name] = [val]

                result = func(*new_args, **new_kwargs)
                return result[0]

        return wrapper

    return decorator
