"""make_heatmap_url_nullable

Revision ID: bafe8d8ad688
Revises: 3c1f4e8a0d5a
Create Date: 2025-07-29 12:45:30.465246

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bafe8d8ad688'
down_revision: Union[str, None] = '3c1f4e8a0d5a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('diagnosis_reports', 'heatmap_url', nullable=True)
    pass


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('diagnosis_reports', 'heatmap_url', nullable=False)
    pass
