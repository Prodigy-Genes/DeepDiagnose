"""Add medical prediction enhancements

Revision ID: 3c1f4e8a0d5a
Revises: 558bd396e190
Create Date: 2025-07-27

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

# revision identifiers
revision = '3c1f4e8a0d5a'
down_revision = '558bd396e190'
branch_labels = None
depends_on = None

def upgrade():
    # Check if modality column exists before dropping
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('medical_images')]
    if 'modality' in columns:
        op.drop_column('medical_images', 'modality')

    # Add new columns to medical_images with conditional checks
    table_name = 'medical_images'
    existing_columns = [col['name'] for col in inspector.get_columns(table_name)]
    
    if 'original_filename' not in existing_columns:
        op.add_column(table_name, sa.Column('original_filename', sa.String(length=255), nullable=True))
    if 'scan_type' not in existing_columns:
        op.add_column(table_name, sa.Column('scan_type', sa.String(length=50), nullable=True))
    if 'scan_type_confidence' not in existing_columns:
        op.add_column(table_name, sa.Column('scan_type_confidence', sa.Float(), nullable=True))
    if 'anatomy' not in existing_columns:
        op.add_column(table_name, sa.Column('anatomy', sa.String(length=100), nullable=True))
    if 'anatomy_confidence' not in existing_columns:
        op.add_column(table_name, sa.Column('anatomy_confidence', sa.Float(), nullable=True))
    if 'disease' not in existing_columns:
        op.add_column(table_name, sa.Column('disease', sa.String(length=100), nullable=True))
    if 'disease_confidence' not in existing_columns:
        op.add_column(table_name, sa.Column('disease_confidence', sa.Float(), nullable=True))
    if 'overlay_image_url' not in existing_columns:
        op.add_column(table_name, sa.Column('overlay_image_url', sa.Text(), nullable=True))
    if 'explanation' not in existing_columns:
        op.add_column(table_name, sa.Column('explanation', sa.Text(), nullable=True))
    if 'prediction_results' not in existing_columns:
        op.add_column(table_name, sa.Column('prediction_results', postgresql.JSONB(), nullable=True))
    if 'processed_at' not in existing_columns:
        op.add_column(table_name, sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True))
    if 'processing_error' not in existing_columns:
        op.add_column(table_name, sa.Column('processing_error', sa.Text(), nullable=True))

    # Rename columns in diagnosis_reports if they exist
    diagnosis_columns = [col['name'] for col in inspector.get_columns('diagnosis_reports')]
    
    if 'diagnosis_result' in diagnosis_columns:
        op.alter_column('diagnosis_reports', 'diagnosis_result', new_column_name='diagnosis_summary')
    if 'confidence_score' in diagnosis_columns:
        op.alter_column('diagnosis_reports', 'confidence_score', new_column_name='overall_confidence')
    
    # Add new columns to diagnosis_reports
    if 'confidence_breakdown' not in diagnosis_columns:
        op.add_column('diagnosis_reports', sa.Column('confidence_breakdown', postgresql.JSONB(), nullable=True))
    if 'recommendations' not in diagnosis_columns:
        op.add_column('diagnosis_reports', sa.Column('recommendations', sa.Text(), nullable=True))
    if 'reviewed' not in diagnosis_columns:
        op.add_column('diagnosis_reports', sa.Column('reviewed', sa.Boolean(), server_default='false', nullable=False))
    if 'review_notes' not in diagnosis_columns:
        op.add_column('diagnosis_reports', sa.Column('review_notes', sa.Text(), nullable=True))
    
    # Update foreign key for CASCADE delete
    op.drop_constraint('diagnosis_reports_image_id_fkey', 'diagnosis_reports', type_='foreignkey')
    op.create_foreign_key('diagnosis_reports_image_id_fkey', 'diagnosis_reports', 'medical_images', 
                          ['image_id'], ['image_id'], ondelete='CASCADE')

    # Add new columns to system_logs
    logs_columns = [col['name'] for col in inspector.get_columns('system_logs')]
    
    if 'user_agent' not in logs_columns:
        op.add_column('system_logs', sa.Column('user_agent', sa.String(length=500), nullable=True))
    if 'resource_id' not in logs_columns:
        op.add_column('system_logs', sa.Column('resource_id', sa.String(length=100), nullable=True))
    if 'resource_type' not in logs_columns:
        op.add_column('system_logs', sa.Column('resource_type', sa.String(length=50), nullable=True))
    if 'status' not in logs_columns:
        op.add_column('system_logs', sa.Column('status', sa.String(length=20), server_default='success', nullable=False))

    # Create reset_codes table only if it doesn't exist
    if not inspector.has_table('reset_codes'):
        op.create_table('reset_codes',
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('code', sa.String(length=6), nullable=False),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
            sa.Column('used', sa.Boolean(), server_default='false', nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('email')
        )

    # Create indexes only if they don't exist
    medical_indexes = [idx['name'] for idx in inspector.get_indexes('medical_images')]
    logs_indexes = [idx['name'] for idx in inspector.get_indexes('system_logs')]
    
    if 'idx_medical_images_user_id' not in medical_indexes:
        op.create_index('idx_medical_images_user_id', 'medical_images', ['user_id'])
    if 'idx_medical_images_processed' not in medical_indexes:
        op.create_index('idx_medical_images_processed', 'medical_images', ['processed'])
    if 'idx_medical_images_disease' not in medical_indexes:
        op.create_index('idx_medical_images_disease', 'medical_images', ['disease'])
    if 'idx_medical_images_uploaded_at' not in medical_indexes:
        op.create_index('idx_medical_images_uploaded_at', 'medical_images', ['uploaded_at'])
    if 'idx_system_logs_user_id' not in logs_indexes:
        op.create_index('idx_system_logs_user_id', 'system_logs', ['user_id'])
    if 'idx_system_logs_timestamp' not in logs_indexes:
        op.create_index('idx_system_logs_timestamp', 'system_logs', ['timestamp'])
    if 'idx_system_logs_action' not in logs_indexes:
        op.create_index('idx_system_logs_action', 'system_logs', ['action'])
        # Add heatmap_url column to diagnosis_reports
    if 'heatmap_url' not in diagnosis_columns:
        op.add_column('diagnosis_reports', sa.Column('heatmap_url', sa.Text(), nullable=True))


def downgrade():
    # Drop indexes
    op.drop_index('idx_system_logs_action', table_name='system_logs')
    op.drop_index('idx_system_logs_timestamp', table_name='system_logs')
    op.drop_index('idx_system_logs_user_id', table_name='system_logs')
    op.drop_index('idx_medical_images_uploaded_at', table_name='medical_images')
    op.drop_index('idx_medical_images_disease', table_name='medical_images')
    op.drop_index('idx_medical_images_processed', table_name='medical_images')
    op.drop_index('idx_medical_images_user_id', table_name='medical_images')
    
    # Drop reset_codes table
    op.drop_table('reset_codes')
    
    # Revert system_logs changes
    op.drop_column('system_logs', 'status')
    op.drop_column('system_logs', 'resource_type')
    op.drop_column('system_logs', 'resource_id')
    op.drop_column('system_logs', 'user_agent')
    
    # Revert diagnosis_reports changes
    op.drop_constraint('diagnosis_reports_image_id_fkey', 'diagnosis_reports', type_='foreignkey')
    op.create_foreign_key('diagnosis_reports_image_id_fkey', 'diagnosis_reports', 'medical_images', ['image_id'], ['image_id'])
    op.drop_column('diagnosis_reports', 'review_notes')
    op.drop_column('diagnosis_reports', 'reviewed')
    op.drop_column('diagnosis_reports', 'recommendations')
    op.drop_column('diagnosis_reports', 'confidence_breakdown')
    op.alter_column('diagnosis_reports', 'diagnosis_summary', new_column_name='diagnosis_result')
    op.alter_column('diagnosis_reports', 'overall_confidence', new_column_name='confidence_score')
    op.drop_column('diagnosis_reports', 'heatmap_url')
    
    # Revert medical_images changes
    op.add_column('medical_images', sa.Column('modality', sa.String(length=20), nullable=False))
    op.drop_column('medical_images', 'processing_error')
    op.drop_column('medical_images', 'processed_at')
    op.drop_column('medical_images', 'prediction_results')
    op.drop_column('medical_images', 'explanation')
    op.drop_column('medical_images', 'overlay_image_url')
    op.drop_column('medical_images', 'disease_confidence')
    op.drop_column('medical_images', 'disease')
    op.drop_column('medical_images', 'anatomy_confidence')
    op.drop_column('medical_images', 'anatomy')
    op.drop_column('medical_images', 'scan_type_confidence')
    op.drop_column('medical_images', 'scan_type')
    op.drop_column('medical_images', 'original_filename')
    