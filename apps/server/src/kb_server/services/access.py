from kb_server.domain.credentials import Principal, ReadAccess
from kb_server.storage.repositories.users import UserRepository


def read_access(principal: Principal, *, users: UserRepository) -> ReadAccess | None:
    """Return what `principal` may read: the one place that decides it.

    Listing, getting and searching documents each translate the result into their
    own query, so they agree on who sees what.

    Args:
        principal: Who asks. A front end acting for someone gets that user's access.
        users: Reads the asker's groups and managed collections.

    Returns:
        None when the asker reads everything (an admin, not acting for anyone);
        otherwise their groups, and the collections they manage if they're a manager.
    """
    if principal.is_admin:
        return None
    group_ids = users.list_group_ids(principal.user.id)
    managed_collection_ids = (
        users.list_managed_collection_ids(principal.user.id)
        if principal.user.role == "manager"
        else set()
    )
    return ReadAccess(
        group_ids=frozenset(group_ids),
        managed_collection_ids=frozenset(managed_collection_ids),
    )
