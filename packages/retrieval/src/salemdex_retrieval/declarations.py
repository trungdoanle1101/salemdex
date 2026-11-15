"""What every declaration in a schema class body shares: a name taken from that body.

`Field` (in `schema.py`) and `Vector` (in `vectors.py`) build on `Declaration`, so
`Schema` can collect both and apply one set of naming rules across them.
"""

from typing import Any, ClassVar

from salemdex_retrieval.errors import SchemaError


class Declaration:
    """Something a schema declares in its class body, named after its attribute.

    A declaration belongs to one schema: declaring the same object under a second name
    is refused. Subclasses set `_kind` ("field", "vector") to the word their messages
    use.
    """

    _kind: ClassVar[str]

    def __init__(self) -> None:
        """Start without a name; the schema class body gives it one."""
        self._name: str | None = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Take the attribute's name as the declaration's name.

        Raises:
            SchemaError: The declaration already has a name, in this class or another.
        """
        if self._name is not None:
            raise SchemaError(
                f"this {type(self).__name__}(...) is already the {self._kind} {self._name!r}; "
                f"declare a new {type(self).__name__}(...) for {name!r} "
                f"in {owner.__name__}"
            )
        self._name = name

    @property
    def name(self) -> str:
        """The name the declaration was declared under.

        Raises:
            SchemaError: The declaration wasn't made in a class body.
        """
        if self._name is None:
            type_name = type(self).__name__
            raise SchemaError(
                f"{type_name} has no name; declare it in a schema class body, "
                f"e.g. my_{self._kind} = {type_name}(...)"
            )
        return self._name
