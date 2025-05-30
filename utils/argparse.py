import ast

def smart_cast(raw: str):
    """Turn a CLI string into int/float/bool/str/list/dict."""
    # Step 1. anything that already *is* a Python literal
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass

    # Step 2. "[a,b,c]"  →  ['a', 'b', 'c']
    if raw.startswith('[') and raw.endswith(']'):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [smart_cast(x.strip()) for x in inner.split(',')]

    # Step 3. "{a:b,c:d}"  →  {'a': 'b', 'c': 'd'}
    if raw.startswith('{') and raw.endswith('}'):
        inner = raw[1:-1].strip()
        if not inner:
            return {}
        d = {}
        for pair in inner.split(','):
            if ':' not in pair:
                break      # malformed → fall through to raw string
            k, v = (x.strip() for x in pair.split(':', 1))
            d[k] = smart_cast(v)
        else:
            return d      # parsed every pair successfully

    # Step 4. booleans / numbers that aren't valid literals for some reason
    low = raw.lower()
    if low in ('true', 'false'):
        return low == 'true'
    try:
        return int(raw) if '.' not in raw else float(raw)
    except ValueError:
        return raw  # final fallback → keep as plain string


def collect_kwargs(unknown: list[str], prefix="") -> dict:
    kwargs = {}
    for arg in unknown:
        if arg.startswith(f'--{prefix}') and '=' in arg:
            key, value = arg.lstrip('-').split('=', 1)
            kwargs[key] = smart_cast(value)
    return kwargs