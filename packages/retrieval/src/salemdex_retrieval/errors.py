"""The errors of a wrongly declared schema, of what doesn't fit one, of bad statements,
and of models that can't be loaded.

They live apart from `schema.py` and `chunks.py` so every module can raise them without
importing the others.
"""


class SchemaError(Exception):
    """Raised when a schema, a field or a vector is declared against the rules."""


class SchemaMismatchError(ValueError):
    """Raised when a chunk, a filter or a stored index doesn't fit its schema.

    A chunk's field values don't match the schema's fields, a filter uses a field
    that isn't one of them, or what a backend holds contradicts the schema (a stored
    point, a payload index of the wrong type). The schema itself is fine (that would be
    a `SchemaError`), so retrying the same call can't help.
    """


class StatementError(ValueError):
    """Raised when a statement's stages come in an order its steps don't allow.

    Only a statement built at run time (from config, by an agent) can be out of order:
    the steps refuse it in code, and `replay` refuses it at run time. An API can answer
    it as a bad request.
    """


class ModelLoadError(Exception):
    """Raised when a model can't be loaded.

    No local folder or hub model has its name, or what is there isn't a model. Each
    calls for the same fix, correcting the model's name, so they share one error; the
    library's own error is chained as the cause and tells them apart. Neither an
    `OSError` nor a `ValueError`, since each would be wrong for some of the causes.
    """
