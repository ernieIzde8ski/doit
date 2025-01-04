from typing import TypeGuard


def is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def is_str_list(value: object) -> TypeGuard[list[str]]:
    return is_object_list(value) and all(isinstance(s, (str)) for s in value)
