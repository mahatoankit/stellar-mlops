"""
Database Connection Utilities for Stellar Classification Pipeline

This module provides database connection and utility functions for the
MariaDB ColumnStore database integration. It follows the "One Big Table"
approach for optimal ML performance.

Author: MLOps Team
Date: September 2025
"""

import os
import logging
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration class that reads from environment variables."""

    def __init__(self):
        self.host = os.getenv("MARIADB_HOST", "localhost")
        self.port = int(os.getenv("MARIADB_PORT", "3306"))
        self.database = os.getenv("MARIADB_DATABASE", "stellar_db")
        self.user = os.getenv("MARIADB_USER", "stellar_user")
        self.password = os.getenv("MARIADB_PASSWORD", "stellar_user_password")

    def get_connection_params(self) -> Dict[str, Any]:
        """Return connection parameters as dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "autocommit": True,
            "charset": "utf8mb4",
        }


@contextmanager
def get_db_connection(config: Optional[DatabaseConfig] = None):
    """
    Context manager for database connections with automatic cleanup.

    Design Rationale:
    - Ensures proper connection cleanup and resource management
    - Provides consistent error handling across the pipeline
    - Supports both transaction and autocommit modes

    Args:
        config: DatabaseConfig instance, creates new if None

    Yields:
        mysql.connector connection object

    Raises:
        mysql.connector.Error: For database connection issues
    """
    if config is None:
        config = DatabaseConfig()

    connection = None
    try:
        connection = mysql.connector.connect(**config.get_connection_params())
        logger.info(f"Successfully connected to MariaDB at {config.host}:{config.port}")
        yield connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")


def test_connection() -> bool:
    """
    Test database connectivity and return status.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result[0] == 1
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


def execute_query(
    query: str, params: Tuple = None, fetch: bool = False
) -> Optional[Any]:
    """
    Execute a SQL query with optional parameters.

    Args:
        query: SQL query string
        params: Query parameters tuple
        fetch: Whether to fetch and return results

    Returns:
        Query results if fetch=True, None otherwise

    Raises:
        mysql.connector.Error: For SQL execution errors
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())

            if fetch:
                results = cursor.fetchall()
                cursor.close()
                return results
            else:
                conn.commit()
                cursor.close()
                return None

    except Error as e:
        logger.error(f"Query execution error: {e}")
        logger.error(f"Query: {query}")
        raise


def execute_many(query: str, data: list) -> int:
    """
    Execute a query with multiple parameter sets (bulk insert).

    Args:
        query: SQL query string with placeholders
        data: List of parameter tuples

    Returns:
        Number of affected rows

    Raises:
        mysql.connector.Error: For SQL execution errors
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, data)
            affected_rows = cursor.rowcount
            conn.commit()
            cursor.close()
            logger.info(f"Bulk operation completed: {affected_rows} rows affected")
            return affected_rows

    except Error as e:
        logger.error(f"Bulk operation error: {e}")
        raise


def dataframe_to_db(
    df: pd.DataFrame, table_name: str, if_exists: str = "append"
) -> int:
    """
    Insert pandas DataFrame into database table.

    Design Rationale:
    - Optimized for MariaDB ColumnStore bulk loading
    - Handles data type conversion automatically
    - Provides progress logging for large datasets

    Args:
        df: pandas DataFrame to insert
        table_name: Target table name
        if_exists: Action if table exists ('append', 'replace', 'fail')

    Returns:
        Number of rows inserted

    Raises:
        ValueError: For invalid parameters
        mysql.connector.Error: For database errors
    """
    if df.empty:
        logger.warning("Attempting to insert empty DataFrame")
        return 0

    config = DatabaseConfig()

    try:
        # Create SQLAlchemy engine for pandas integration
        from sqlalchemy import create_engine

        connection_string = (
            f"mysql+mysqlconnector://{config.user}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )

        engine = create_engine(connection_string)

        # Insert DataFrame to database
        rows_inserted = df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=1000,  # Optimize for ColumnStore
        )

        logger.info(f"Successfully inserted {len(df)} rows into {table_name}")
        return len(df)

    except Exception as e:
        logger.error(f"DataFrame insertion error: {e}")
        raise


def query_to_dataframe(query: str, params: Tuple = None) -> pd.DataFrame:
    """
    Execute query and return results as pandas DataFrame.

    Args:
        query: SQL query string
        params: Query parameters tuple

    Returns:
        pandas DataFrame with query results

    Raises:
        mysql.connector.Error: For SQL execution errors
    """
    try:
        config = DatabaseConfig()

        # Create SQLAlchemy engine for pandas integration
        from sqlalchemy import create_engine

        connection_string = (
            f"mysql+mysqlconnector://{config.user}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )

        engine = create_engine(connection_string)

        # Execute query and return DataFrame
        df = pd.read_sql(query, engine, params=params)

        logger.info(f"Query returned {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Query to DataFrame error: {e}")
        raise


def get_table_info(table_name: str) -> Dict[str, Any]:
    """
    Get metadata information about a table.

    Args:
        table_name: Name of the table to inspect

    Returns:
        Dictionary with table metadata (row count, columns, etc.)
    """
    try:
        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        count_result = execute_query(count_query, fetch=True)
        row_count = count_result[0]["row_count"] if count_result else 0

        # Get column information
        columns_query = f"DESCRIBE {table_name}"
        columns_info = execute_query(columns_query, fetch=True)

        return {
            "table_name": table_name,
            "row_count": row_count,
            "columns": columns_info,
        }

    except Exception as e:
        logger.error(f"Error getting table info for {table_name}: {e}")
        raise


# Initialize module
if __name__ == "__main__":
    # Test connection when module is run directly
    print("Testing database connection...")
    if test_connection():
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed!")
