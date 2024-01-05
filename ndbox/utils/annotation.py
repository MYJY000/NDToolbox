def base_annotation(annotation):
    def decorator_annotation(**kwargs):
        def decorator(func):
            sub_annotation_name = annotation.__name__
            if func.__annotations__.get(sub_annotation_name) is None:
                func.__annotations__[sub_annotation_name] = {}
            annotation(func, *kwargs)
            return func
        return decorator
    return decorator_annotation


@base_annotation
def anno1(func, **kwargs):
    for key, value in kwargs.items():
        func.__annotations__['anno1'][key] = value


def add_annotation(**kwargs):
    def decorator(func):
        for key, value in kwargs.items():
            func.__annotations__[key] = value
        return func
    return decorator
