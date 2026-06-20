import os
import yaml
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data_for_analytics(config_path="config/config.yaml"):
    """
    Reads processed parquet files (using pandas), performs aggregations,
    and saves lightweight datasets specifically for the Streamlit dashboard.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processed_dir = config['directories']['processed_data']
    analytics_dir = config['directories']['analytics_data']
    
    os.makedirs(analytics_dir, exist_ok=True)

    logging.info("⏳ Loading Parquet data for Analytics Processing...")
    
    # We load columns we actually need to save memory
    train_df = pd.read_parquet(os.path.join(processed_dir, 'train.parquet'))
    members_df = pd.read_parquet(os.path.join(processed_dir, 'members.parquet'))
    
    # 1. Demographics & Churn Summary
    logging.info("⚙️  Creating Demographics summary...")
    demo_df = train_df.merge(members_df, on='msno', how='inner')
    
    import numpy as np
    # Clean up demographics
    demo_df['bd'] = demo_df['bd'].apply(lambda x: x if pd.notnull(x) and 0 < x < 100 else np.nan) # filter out impossible ages
    
    # Create Age Groups
    bins = [0, 18, 25, 35, 45, 55, 100]
    labels = ['<18', '18-25', '26-35', '36-45', '46-55', '55+']
    demo_df['age_group'] = pd.cut(demo_df['bd'], bins=bins, labels=labels)
    
    demo_summary = demo_df.to_parquet(os.path.join(analytics_dir, 'demographics_churn.parquet'), index=False)
    
    # Free up memory
    del demo_df
    
    # 2. Transactions Summary (Promo ratio, Revenue)
    logging.info("⚙️  Processing Transactions...")
    
    # Read transactions in chunks if it's too large, or load just what we need
    # We need: msno, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, is_cancel
    cols_to_use = ['msno', 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_cancel', 'transaction_date']
    trans_df = pd.read_parquet(os.path.join(processed_dir, 'transactions.parquet'), columns=cols_to_use)
    
    # Identify promos (where paid is 0 but list price > 0, or just paid is 0)
    trans_df['is_promo'] = (trans_df['actual_amount_paid'] == 0).astype(int)
    trans_df['revenue_lost'] = trans_df['plan_list_price'] - trans_df['actual_amount_paid']
    # Filter negative revenue lost (errors in data)
    trans_df['revenue_lost'] = trans_df['revenue_lost'].apply(lambda x: x if x > 0 else 0)
    
    # Merge with churn to see promo usage among churned vs retained
    trans_churn = trans_df.merge(train_df[['msno', 'is_churn']], on='msno', how='inner')
    
    # Aggregate transaction KPIs per user
    user_trans_kpis = trans_churn.groupby('msno').agg(
        total_revenue=('actual_amount_paid', 'sum'),
        total_revenue_lost=('revenue_lost', 'sum'),
        promo_count=('is_promo', 'sum'),
        cancel_count=('is_cancel', 'sum'),
        is_churn=('is_churn', 'first')
    ).reset_index()
    
    user_trans_kpis.to_parquet(os.path.join(analytics_dir, 'user_transaction_kpis.parquet'), index=False)
    
    # Free up memory
    del trans_df
    del trans_churn
    
    # 3. User Logs Summary (Engagement)
    logging.info("⚙️  Processing User Logs...")
    
    cols_logs = ['msno', 'date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']
    logs_df = pd.read_parquet(os.path.join(processed_dir, 'user_logs.parquet'), columns=cols_logs)
    
    # Aggregate engagement
    logs_df['total_songs_played'] = logs_df['num_25'] + logs_df['num_50'] + logs_df['num_75'] + logs_df['num_985'] + logs_df['num_100']
    
    user_engagement = logs_df.groupby('msno').agg(
        active_days=('date', 'count'),
        total_secs=('total_secs', 'sum'),
        songs_played=('total_songs_played', 'sum'),
        unique_songs=('num_unq', 'sum'),
        songs_completed=('num_100', 'sum')
    ).reset_index()
    
    # Merge with Churn label
    user_engagement = user_engagement.merge(train_df[['msno', 'is_churn']], on='msno', how='inner')
    user_engagement.to_parquet(os.path.join(analytics_dir, 'user_engagement.parquet'), index=False)
    
    # 4. Global KPIs
    logging.info("⚙️  Calculating Global KPIs...")
    total_users = len(train_df)
    churned_users = train_df['is_churn'].sum()
    churn_rate = churned_users / total_users
    
    global_kpis = pd.DataFrame({
        'total_users': [total_users],
        'churned_users': [churned_users],
        'churn_rate': [churn_rate]
    })
    global_kpis.to_csv(os.path.join(analytics_dir, 'global_kpis.csv'), index=False)
    
    logging.info(f"✅ Dashboard Data successfully created in {analytics_dir}")

if __name__ == "__main__":
    process_data_for_analytics()
