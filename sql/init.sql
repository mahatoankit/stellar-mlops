-- =============================================================================
-- STELLAR CLASSIFICATION DATABASE INITIALIZATION
-- =============================================================================
-- This script creates the initial database schema for the stellar classification
-- pipeline using the "One Big Table" approach for optimal ML performance.
--
-- Design Rationale:
-- - Single table design eliminates JOINs for ML data retrieval
-- - MariaDB ColumnStore optimized for analytical queries on large datasets
-- - All features and target variable co-located for efficient training data access
-- =============================================================================

USE stellar_db;

-- Create the main stellar_data table using ColumnStore engine
-- This table follows the "One Big Table" approach where all features
-- and metadata are stored together for optimal ML performance
CREATE TABLE IF NOT EXISTS stellar_data (
    -- Primary key and metadata
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    obj_ID BIGINT NOT NULL COMMENT 'Object identifier from SDSS',
    
    -- Celestial coordinates
    alpha DOUBLE COMMENT 'Right ascension angle (at J2000 epoch)',
    delta_coord DOUBLE COMMENT 'Declination angle (at J2000 epoch)', -- 'delta' is reserved keyword
    
    -- Photometric magnitudes (5-band SDSS system)
    u DOUBLE COMMENT 'Ultraviolet filter magnitude',
    g DOUBLE COMMENT 'Green filter magnitude', 
    r DOUBLE COMMENT 'Red filter magnitude',
    i DOUBLE COMMENT 'Near infrared filter magnitude',
    z DOUBLE COMMENT 'Infrared filter magnitude',
    
    -- Spectroscopic data
    run_ID INT COMMENT 'Run number used to identify the specific scan',
    rerun_ID INT COMMENT 'Rerun number to specify how the image was processed',
    cam_col INT COMMENT 'Camera column to identify the scanline within the run',
    field_ID INT COMMENT 'Field number to identify each field',
    spec_obj_ID BIGINT COMMENT 'Unique ID used for optical spectroscopic objects',
    class VARCHAR(10) COMMENT 'Object class (STAR, GALAXY, QSO)',
    redshift DOUBLE COMMENT 'Redshift value based on increase in wavelength',
    plate INT COMMENT 'Plate ID for spectroscopic identification',
    MJD INT COMMENT 'Modified Julian Date for observation timing',
    fiber_ID INT COMMENT 'Fiber ID for spectroscopic identification',
    
    -- Processing metadata
    is_processed BOOLEAN DEFAULT FALSE COMMENT 'Flag indicating if record has been preprocessed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Record creation timestamp',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    
    -- Engineered features (populated during preprocessing)
    u_g_color DOUBLE COMMENT 'Color index: u-g magnitude difference',
    g_r_color DOUBLE COMMENT 'Color index: g-r magnitude difference', 
    r_i_color DOUBLE COMMENT 'Color index: r-i magnitude difference',
    i_z_color DOUBLE COMMENT 'Color index: i-z magnitude difference',
    
    -- Scaling and normalization flags
    is_scaled BOOLEAN DEFAULT FALSE COMMENT 'Flag indicating if features have been scaled',
    data_split VARCHAR(10) COMMENT 'Data split assignment: TRAIN, TEST, VALIDATION',
    
    -- Indexing for performance
    INDEX idx_class (class),
    INDEX idx_processed (is_processed),
    INDEX idx_split (data_split),
    INDEX idx_obj_id (obj_ID),
    INDEX idx_created (created_at)
    
) ENGINE=ColumnStore COMMENT='Stellar classification data - One Big Table approach for ML optimization';

-- Create a view for easy access to processed training data
CREATE OR REPLACE VIEW training_data_view AS
SELECT 
    obj_ID,
    alpha, delta_coord,
    u, g, r, i, z,
    u_g_color, g_r_color, r_i_color, i_z_color,
    redshift, plate, MJD, fiber_ID,
    class,
    data_split
FROM stellar_data 
WHERE is_processed = TRUE;

-- Create a view for feature engineering
CREATE OR REPLACE VIEW feature_engineering_view AS
SELECT 
    id,
    obj_ID,
    u, g, r, i, z,
    (u - g) as calculated_u_g,
    (g - r) as calculated_g_r,
    (r - i) as calculated_r_i,
    (i - z) as calculated_i_z,
    redshift,
    class,
    is_processed
FROM stellar_data 
WHERE is_processed = FALSE;

-- Add some initial metadata for tracking pipeline runs
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR(50) PRIMARY KEY,
    run_type ENUM('INGESTION', 'PREPROCESSING', 'TRAINING', 'INFERENCE') NOT NULL,
    status ENUM('STARTED', 'COMPLETED', 'FAILED') NOT NULL,
    records_processed INT DEFAULT 0,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    error_message TEXT NULL,
    metadata JSON NULL COMMENT 'Additional run metadata'
) ENGINE=InnoDB;

-- Insert initial pipeline run record
INSERT IGNORE INTO pipeline_runs (run_id, run_type, status) 
VALUES ('INIT_001', 'INGESTION', 'STARTED');

-- Performance optimization: Analyze table for ColumnStore
-- This helps ColumnStore optimize query performance
ANALYZE TABLE stellar_data;

COMMIT;
