def get_parent_module(obj):
    module_tree = getattr(obj, "__module__", None)
    return module_tree.split(".")[0] if module_tree else None


def simplify(obj):
    if get_parent_module(obj) == "ramanujantools":
        return obj.simplify()
    else:
        import sympy as sp

        return sp.simplify(obj)
