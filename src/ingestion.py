import os
import yaml
import pandas as pd
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data_with_pandas(config_path="config/config.yaml"):
    """
    Reads raw CSVs from data/raw using pandas chunking, 
    converts them to Parquet, and saves them to data/processed.
    """
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['directories']['raw_data']
    processed_dir = config['directories']['processed_data']
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Map friendly names to actual filenames from config
    files = {
        'train': config['files']['train'],
        'members': config['files']['members'],
        'transactions': config['files']['transactions'],
        'user_logs': config['files']['user_logs']
    }

    # Data Types to minimize memory usage
    dtype_options = {
        'gender': 'category', 
        'msno': 'object',
        'payment_method_id': 'Int32',
        'city': 'Int32',
        'registered_via': 'Int32'
    }

    chunk_size = 500_000  # Process 500k rows at a time to save memory

    # 2. Process each file
    for key, filename in files.items():
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(processed_dir, f"{key}.parquet")
        
        if not os.path.exists(input_path):
            logging.error(f"❌ Input file not found: {input_path}")
            continue

        logging.info(f"🚀 Processing {key} ({filename}) in chunks...")

        try:
            # We will read in chunks, convert to parquet and append or write whole if we collect it
            # To be efficient, we can process and write each chunk. But fastparquet/pyarrow with pandas
            # makes appending sometimes tricky if we don't do it right.
            # An easier way is to collect the chunks into a list, concat, then write if it fits in memory,
            # or save chunks to disk and then read them as a single partitioned dataset.
            # Given we want simple pandas, let's process chunks and append to a parquet file using PyArrow.
            
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            writer = None
            
            for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, dtype=dtype_options, low_memory=False)):
                # Standardize columns
                chunk.columns = [c.lower().strip() for c in chunk.columns]
                
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    # Initialize writer on first chunk
                    writer = pq.ParquetWriter(output_path, table.schema, compression='zstd')
                
                writer.write_table(table)
                
                if i % 10 == 0:
                    logging.info(f"   ...processed {i * chunk_size} rows for {key}")

            if writer:
                writer.close()
            
            logging.info(f"✅ Successfully converted {key} to Parquet at {output_path}.")

        except Exception as e:
            logging.error(f"❌ Failed to ingest {key}: {e}")
            raise e

if __name__ == "__main__":
    ingest_data_with_pandas()