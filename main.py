import logging
import argparse
from src.ingestion import ingest_data_with_pandas
from src.data_processing import process_data_for_analytics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(step):
    """
    Main execution entry point for the Data Analytics Pipeline.
    """
    try:
        if step == "all" or step == "ingest":
            logging.info(">>>>>> STAGE 1: Data Ingestion (CSV -> Parquet via Pandas Chunking) <<<<<<")
            ingest_data_with_pandas()
            
        if step == "all" or step == "process":
            logging.info(">>>>>> STAGE 2: Data Processing & KPIs (Pandas Aggregation) <<<<<<")
            process_data_for_analytics()
            
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["ingest", "process", "all"], default="all",
                        help="Which step of the pipeline to run.")
    args = parser.parse_args()
    
    main(args.step)