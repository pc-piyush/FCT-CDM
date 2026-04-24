import duckdb
import os
import pandas as pd

OMOP_TABLES = [
    "CONDITION_OCCURRENCE",
    "DRUG_EXPOSURE",
    "MEASUREMENT",
    "PROCEDURE_OCCURRENCE",
    "VISIT_OCCURRENCE",
    "PERSON"
]




# def ingest_csv_to_duckdb(csv_path, db_path, parquet_path):
#     con = duckdb.connect(db_path)

#     for table in OMOP_TABLES:
#         file = os.path.join(csv_path, f"Deid_{table}_v202601.csv")
#         #Deid_PERSON_v202601

#         print(f"Loading {table}...")

#         con.execute(f"""
#             CREATE OR REPLACE TABLE {table} AS
#             SELECT * FROM read_csv_auto('{file}', all_varchar=True)
#         """)

#         con.execute(f"""
#             COPY {table} TO '{parquet_path}/{table}.parquet' (FORMAT PARQUET)
#         """)

#     con.close()

# def ingest_csv_duckdb_native(csv_path, db_path, parquet_path):
#     con = duckdb.connect(db_path)

#     for table in OMOP_TABLES:
#         file = os.path.join(csv_path, f"Deid_{table}_v202601.csv")

#         print(f" Processing {table} (DuckDB streaming)...")

#         con.execute(f"""
#             CREATE OR REPLACE TABLE {table} AS
#             SELECT * FROM read_csv_auto(
#                 '{file}',
#                 SAMPLE_SIZE=100000,
#                 ALL_VARCHAR=TRUE
#             )
#         """)

#         print(f" Writing {table} → Parquet")

#         con.execute(f"""
#             COPY {table}
#             TO '{parquet_path}/{table}.parquet'
#             (FORMAT PARQUET)
#         """)

#     con.close()


# def ingest_csv_duckdb_native(csv_path, db_path, parquet_path):
#     """
#     Ingest CSV files into DuckDB and export to Parquet.
#     Optimized for large files (handles 263+ GB measurement files).
#     """
    
#     # Create output directory if it doesn't exist
#     os.makedirs(parquet_path, exist_ok=True)
    
#     # Connect to DuckDB with optimized settings for large files
#     con = duckdb.connect(db_path)
    
#     # Configure DuckDB for large file processing
#     con.execute("SET memory_limit='16GB'")  # Adjust based on your system RAM
#     con.execute("SET threads TO 8")  # Adjust based on your CPU cores
#     con.execute("SET temp_directory='/tmp/duckdb_temp'")  # Use fast storage if available
    
#     stats = []
    
#     # Get all CSV files
#     csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    
#     if not csv_files:
#         print(f"⚠️  No CSV files found in {csv_path}")
#         return []
    
#     print(f"📂 Found {len(csv_files)} CSV files to process\n")
#     print("="*70)
    
#     for csv_file in sorted(csv_files):
#         file_path = os.path.join(csv_path, csv_file)
#         table = csv_file.replace('.csv', '').replace('Deid_','').lower()
        
#         print(f"📊 Processing: {csv_file}")
#         print(f"   Table: {table}")
        
#         # Get file size for progress indication
#         file_size_gb = os.path.getsize(file_path) / (1024**3)
#         print(f"   Size: {file_size_gb:.2f} GB")
        
#         try:
#             parquet_file = os.path.join(parquet_path, f"{table}.parquet")
            
#             # For large files (>20GB), use streaming approach
#             if file_size_gb > 20:
#                 print(f"🔄 Using optimized mode for very large file...")
#                 print(f"📥 Creating table in DuckDB (this may take a while)...")
                
#                 # DuckDB can handle this even if larger than RAM by using disk
#                 con.execute(f"""
#                     CREATE OR REPLACE TABLE {table} AS
#                     SELECT * FROM read_csv_auto(
#                         '{file_path}',
#                         SAMPLE_SIZE=1000000,
#                         ALL_VARCHAR=TRUE,
#                         HEADER=TRUE,
#                         IGNORE_ERRORS=TRUE,
#                         DATEFORMAT='%Y-%m-%d',
#                         TIMESTAMPFORMAT='%Y-%m-%d %H:%M:%S',
#                         PARALLEL=TRUE,
#                         BUFFER_SIZE=2097152
#                     )
#                 """)
                
#                 # Get row count from the parquet file (much faster)
#                 print(f"📈 Counting rows from Parquet...")
#                 result = con.execute(f"SELECT COUNT(*) FROM '{parquet_file}'").fetchone()
#                 row_count = result[0] if result else 0
                
#                 # Try to get unique patients if person_id column exists
#                 try:
#                     unique_patients = con.execute(f"""
#                         SELECT COUNT(DISTINCT person_id) 
#                         FROM '{parquet_file}'
#                     """).fetchone()[0]
#                 except:
#                     unique_patients = 'N/A'
                
#             else:
#                 # For smaller files, use the original approach
#                 print(f"📥 Loading into DuckDB...")
#                 con.execute(f"""
#                     CREATE OR REPLACE TABLE {table} AS
#                     SELECT * FROM read_csv_auto(
#                         '{file_path}',
#                         SAMPLE_SIZE=100000,
#                         ALL_VARCHAR=TRUE,
#                         HEADER=TRUE,
#                         IGNORE_ERRORS=TRUE,
#                         DATEFORMAT='%Y-%m-%d',
#                         TIMESTAMPFORMAT='%Y-%m-%d %H:%M:%S'
#                     )
#                 """)
                
#                 # Get statistics
#                 result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
#                 row_count = result[0] if result else 0
                
#                 # Try to get unique patients
#                 try:
#                     unique_patients = con.execute(f"""
#                         SELECT COUNT(DISTINCT person_id) FROM {table}
#                     """).fetchone()[0]
#                 except:
#                     unique_patients = 'N/A'
                
#                 print(f" ✓ Loaded {row_count:,} rows")
#                 if unique_patients != 'N/A':
#                     print(f" ✓ Unique patients: {unique_patients:,}")
                
#                 # Export to Parquet
#                 print(f"💾 Writing to Parquet...")
#                 con.execute(f"""
#                     COPY {table}
#                     TO '{parquet_file}'
#                     (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
#                 """)
            
#             # Get final file size
#             if os.path.exists(parquet_file):
#                 size_mb = os.path.getsize(parquet_file) / (1024**2)
#                 compression_ratio = (file_size_gb * 1024) / size_mb
#                 print(f" ✓ Output: {size_mb:.2f} MB")
#                 print(f" ✓ Compression: {compression_ratio:.1f}x")
#                 print(f" ✓ Rows: {row_count:,}")
#                 if unique_patients != 'N/A':
#                     print(f" ✓ Unique patients: {unique_patients:,}")
#                 print()
                
#                 stats.append({
#                     'table': table,
#                     'rows': row_count,
#                     'patients': unique_patients,
#                     'size_mb': size_mb,
#                     'original_gb': file_size_gb
#                 })
            
#         except Exception as e:
#             print(f" ✗ Error: {str(e)}\n")
#             continue
    
#     # Print summary
#     if stats:
#         print("="*70)
#         print("INGESTION SUMMARY")
#         print("="*70)
#         stats_df = pd.DataFrame(stats)
#         print(stats_df.to_string(index=False))
#         print(f"\nTotal Parquet size: {stats_df['size_mb'].sum():.2f} MB ({stats_df['size_mb'].sum()/1024:.2f} GB)")
#         print(f"Total original size: {stats_df['original_gb'].sum():.2f} GB")
#         print(f"Total rows: {stats_df['rows'].sum():,}")
#         print(f"Overall compression: {(stats_df['original_gb'].sum() * 1024) / stats_df['size_mb'].sum():.1f}x")
    
#     con.close()
#     print("\n✅ Ingestion complete!")
#     return stats


def ingest_csv_duckdb_native(csv_path, db_path, parquet_path):
    """
    Ingest CSV files into DuckDB and export to Parquet.
    Optimized for large files (handles 263+ GB measurement files).
    ALL files are loaded into DuckDB tables, regardless of size.
    """

    
    # Create output directory if it doesn't exist
    os.makedirs(parquet_path, exist_ok=True)
    
    # Connect to DuckDB with optimized settings for large files
    con = duckdb.connect(db_path)
    
    # Configure DuckDB for large file processing
    # These settings allow DuckDB to handle data larger than RAM
    con.execute("SET memory_limit='50GB'")  # Adjust based on your system RAM
    con.execute("SET threads TO 8")  # Adjust based on your CPU cores
    con.execute("SET temp_directory='/tmp/duckdb_temp'")  # Use fast storage
    con.execute("SET max_temp_directory_size='500GB'")  # Allow large temp files
    con.execute("SET preserve_insertion_order=false")  # Better performance
    
    stats = []
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.parquet')]
    
    if not csv_files:
        print(f"⚠️  No CSV files found in {csv_path}")
        return []
    
    print(f"📂 Found {len(csv_files)} CSV files to process\n")
    print("="*70)
    
    for csv_file in sorted(csv_files):
        file_path = os.path.join(csv_path, csv_file)
        table = csv_file.replace('.parquet', '').replace('Deid_', '').replace('_v202601', '').lower()
        
        print(f"📊 Processing: {csv_file}")
        print(f"   Table: {table}")
        
        # Get file size for progress indication
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        print(f"   Size: {file_size_gb:.2f} GB")
        
        try:
            # For VERY large files (>50GB), use optimized settings
            if file_size_gb > 50:
                print(f"🔄 Using optimized mode for very large file...")
                print(f"📥 Creating table in DuckDB (this may take a while)...")
                
                # DuckDB can handle this even if larger than RAM by using disk
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table} AS
                    SELECT * FROM read_csv_auto(
                        '{file_path}',
                        SAMPLE_SIZE=1000000,
                        ALL_VARCHAR=TRUE,
                        HEADER=TRUE,
                        IGNORE_ERRORS=TRUE,
                        DATEFORMAT='%Y-%m-%d',
                        TIMESTAMPFORMAT='%Y-%m-%d %H:%M:%S',
                        PARALLEL=TRUE,
                        BUFFER_SIZE=2097152
                    )
                """)
                
            # For large files (10-50GB)
            elif file_size_gb > 20:
                print(f"🔄 Using optimized mode for large file...")
                print(f"📥 Creating table in DuckDB...")
                
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table} AS
                    SELECT * FROM read_csv_auto(
                        '{file_path}',
                        SAMPLE_SIZE=500000,
                        ALL_VARCHAR=TRUE,
                        HEADER=TRUE,
                        IGNORE_ERRORS=TRUE,
                        DATEFORMAT='%Y-%m-%d',
                        TIMESTAMPFORMAT='%Y-%m-%d %H:%M:%S',
                        PARALLEL=TRUE,
                        BUFFER_SIZE=1048576
                    )
                """)
                
            else:
                # For smaller files, use standard approach
                print(f"📥 Loading into DuckDB...")
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table} AS
                    SELECT * FROM read_csv_auto(
                        '{file_path}',
                        SAMPLE_SIZE=100000,
                        ALL_VARCHAR=TRUE,
                        HEADER=TRUE,
                        IGNORE_ERRORS=TRUE,
                        DATEFORMAT='%Y-%m-%d',
                        TIMESTAMPFORMAT='%Y-%m-%d %H:%M:%S'
                    )
                """)
            
            # Get statistics from the DuckDB table
            print(f"📈 Collecting statistics...")
            result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            row_count = result[0] if result else 0
            
            # Try to get unique patients
            try:
                unique_patients = con.execute(f"""
                    SELECT COUNT(DISTINCT person_id) FROM {table}
                """).fetchone()[0]
            except:
                unique_patients = 'N/A'
            
            print(f" ✓ Loaded {row_count:,} rows into DuckDB table '{table}'")
            if unique_patients != 'N/A':
                print(f" ✓ Unique patients: {unique_patients:,}")
            
            # Export to Parquet
            parquet_file = os.path.join(parquet_path, f"{table}.parquet")
            print(f"💾 Exporting to Parquet...")
            
            con.execute(f"""
                COPY {table}
                TO '{parquet_file}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """)
            
            # Get final file size
            if os.path.exists(parquet_file):
                size_mb = os.path.getsize(parquet_file) / (1024**2)
                compression_ratio = (file_size_gb * 1024) / size_mb if size_mb > 0 else 0
                print(f" ✓ Parquet: {size_mb:.2f} MB")
                print(f" ✓ Compression: {compression_ratio:.1f}x")
                print()
                
                stats.append({
                    'table': table,
                    'rows': row_count,
                    'patients': unique_patients,
                    'size_mb': size_mb,
                    'original_gb': file_size_gb
                })
            
        except Exception as e:
            print(f" ✗ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    if stats:
        print("="*70)
        print("INGESTION SUMMARY")
        print("="*70)
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))
        print(f"\nTotal Parquet size: {stats_df['size_mb'].sum():.2f} MB ({stats_df['size_mb'].sum()/1024:.2f} GB)")
        print(f"Total original size: {stats_df['original_gb'].sum():.2f} GB")
        print(f"Total rows: {stats_df['rows'].sum():,}")
        if stats_df['size_mb'].sum() > 0:
            print(f"Overall compression: {(stats_df['original_gb'].sum() * 1024) / stats_df['size_mb'].sum():.1f}x")
    
    # Show all tables in database
    print("\n" + "="*70)
    print("TABLES IN DUCKDB DATABASE")
    print("="*70)
    tables_df = con.execute("SHOW TABLES").df()
    print(tables_df.to_string(index=False))
    
    con.close()
    print("\n✅ Ingestion complete!")
    print(f"✅ All tables loaded into: {db_path}")
    return stats

# Display the corrected function
print("✅ Updated function created!")
print("\nKey changes:")
print("1. ALL files are now loaded into DuckDB tables (no streaming bypass)")
print("2. DuckDB uses disk-based operations for data larger than RAM")
print("3. Optimized settings scale with file size")
print("4. Function shows all tables at the end to confirm they're loaded")

# ingest_csv_to_duckdb("/lustre/blue2/mei.liu/pc/IDR Covid-19 OMOP Non_Human Dataset/Covid_OMOP_dataset_v202601_zipped/clin/", "data/duckdb/omop.db", "data/parquet")
ingest_csv_duckdb_native("/lustre/blue2/mei.liu/pc/IDR Covid-19 OMOP Non_Human Dataset/Covid_OMOP_dataset_v202601_zipped/clin/", "data/duckdb/omop.db", "data/parquet")