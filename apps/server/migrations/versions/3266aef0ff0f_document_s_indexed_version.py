"""Document's indexed_version

Revision ID: 3266aef0ff0f
Revises: d546acf72172
Create Date: 2026-11-19 16:34:42.184757

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3266aef0ff0f"
down_revision: str | Sequence[str] | None = "d546acf72172"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add documents.indexed_version: the version search shows, NULL until first indexed."""
    op.add_column(
        "documents", sa.Column("indexed_version", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """Drop documents.indexed_version."""
    op.drop_column("documents", "indexed_version")
