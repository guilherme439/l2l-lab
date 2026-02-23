import dataclasses
from typing import Union, get_type_hints, get_origin, get_args


def _resolve_dataclass_type(tp):
    if dataclasses.is_dataclass(tp):
        return tp

    if get_origin(tp) is Union:
        for arg in get_args(tp):
            if arg is not type(None) and dataclasses.is_dataclass(arg):
                return arg

    return None


def dataclass_from_dict(cls, data: dict):
    if data is None:
        return cls()

    hints = get_type_hints(cls)
    kwargs = {}

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]
        target = _resolve_dataclass_type(hints[f.name])

        if isinstance(value, dict) and target is not None:
            kwargs[f.name] = dataclass_from_dict(target, value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)
