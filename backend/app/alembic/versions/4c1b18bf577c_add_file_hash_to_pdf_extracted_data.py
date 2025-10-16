"""add_file_hash_to_pdf_extracted_data

Revision ID: 4c1b18bf577c
Revises: 975d96516e1a
Create Date: 2025-10-16 18:50:11.279104

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '4c1b18bf577c'
down_revision = '975d96516e1a'
branch_labels = None
depends_on = None


def upgrade():
    # Add file_hash column
    op.add_column('pdf_extracted_data', sa.Column('file_hash', sa.String(length=64), nullable=True))
    
    # Set unique hash for existing records using MD5(id::text) to ensure uniqueness
    # This is a placeholder since we don't have the original file content
    op.execute("""
        UPDATE pdf_extracted_data 
        SET file_hash = MD5(id::text || filename || created_at) 
        WHERE file_hash IS NULL
    """)
    
    # Make column non-nullable
    op.alter_column('pdf_extracted_data', 'file_hash', nullable=False)
    
    # Create index on file_hash
    op.create_index('ix_pdf_extracted_data_file_hash', 'pdf_extracted_data', ['file_hash'])
    
    # Create unique constraint on owner_id + file_hash to prevent duplicate uploads
    op.create_unique_constraint('uq_pdf_extracted_data_owner_file_hash', 'pdf_extracted_data', ['owner_id', 'file_hash'])


def downgrade():
    # Drop unique constraint
    op.drop_constraint('uq_pdf_extracted_data_owner_file_hash', 'pdf_extracted_data', type_='unique')
    
    # Drop index
    op.drop_index('ix_pdf_extracted_data_file_hash', 'pdf_extracted_data')
    
    # Drop column
    op.drop_column('pdf_extracted_data', 'file_hash')
