"""
Full Analytics Generator for KKBox Churn Dashboard
====================================================
Generates comprehensive analytics datasets for the Streamlit dashboard.
Run with: python src/generate_full_analytics.py
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
ANALYTICS_DIR = "data/analytics"

# Payment method labels (known mappings for KKBox)
PAYMENT_METHOD_LABELS = {
    41: "Credit Card (41)",
    39: "Credit Card (39)",
    38: "Credit Card (38)",
    36: "ATM/Bank Transfer",
    32: "PayPal",
    40: "Online Banking",
    37: "Convenience Store",
    34: "Mobile Payment",
    29: "Gift Card",
    30: "Other"
}

CITY_LABELS = {
    1: "City 1 (Largest)",
    3: "City 3",
    4: "City 4",
    5: "City 5",
    6: "City 6",
    7: "City 7",
    8: "City 8",
    9: "City 9",
    10: "City 10",
    11: "City 11",
    12: "City 12",
    13: "City 13",
    14: "City 14",
    15: "City 15",
    16: "City 16",
    17: "City 17",
    18: "City 18",
    19: "City 19",
    20: "City 20",
    21: "City 21",
    22: "City 22"
}

REG_METHOD_LABELS = {
    3: "iOS App",
    4: "Android App",
    7: "Web Browser",
    9: "Carrier Bundle",
    11: "Third-Party",
    13: "Campaign Link"
}


def save(df, name):
    path = os.path.join(ANALYTICS_DIR, name)
    if name.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    log.info(f"  ✅ Saved {name} → {df.shape}")


def load_base_data():
    log.info("📂 Loading base processed data...")
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, 'train.parquet'))
    members = pd.read_parquet(os.path.join(PROCESSED_DIR, 'members.parquet'))
    transactions = pd.read_parquet(os.path.join(PROCESSED_DIR, 'transactions.parquet'))
    return train, members, transactions


def process_members(members, train):
    """Clean and enrich member data."""
    log.info("👤 Processing member demographics...")
    m = members.copy()

    # Clean birth year → age
    m['bd'] = m['bd'].apply(lambda x: x if pd.notnull(x) and 0 < x < 100 else np.nan)

    bins = [0, 18, 25, 35, 45, 55, 100]
    labels = ['<18', '18-25', '26-35', '36-45', '46-55', '55+']
    m['age_group'] = pd.cut(m['bd'], bins=bins, labels=labels)

    # Registration date features
    m['registration_init_time'] = pd.to_datetime(
        m['registration_init_time'].astype(str), format='%Y%m%d', errors='coerce'
    )
    m['reg_year'] = m['registration_init_time'].dt.year
    m['reg_month'] = m['registration_init_time'].dt.month
    m['reg_quarter'] = m['registration_init_time'].dt.quarter
    m['reg_year_month'] = m['registration_init_time'].dt.to_period('M').astype(str)

    # Labels
    m['city_label'] = m['city'].map(CITY_LABELS).fillna(m['city'].astype(str))
    m['reg_method_label'] = m['registered_via'].map(REG_METHOD_LABELS).fillna('Other')

    # Merge with churn
    merged = train.merge(m, on='msno', how='inner')
    return m, merged


def generate_global_kpis(train, merged_demo):
    """Generate top-level global KPIs."""
    log.info("📊 Generating Global KPIs...")

    total_users = len(train)
    churned = train['is_churn'].sum()
    retained = total_users - churned

    # With gender
    male_churn = merged_demo[merged_demo['gender'] == 'male']['is_churn'].mean() if 'gender' in merged_demo.columns else 0
    female_churn = merged_demo[merged_demo['gender'] == 'female']['is_churn'].mean() if 'gender' in merged_demo.columns else 0

    kpis = pd.DataFrame({
        'total_users': [total_users],
        'churned_users': [churned],
        'retained_users': [retained],
        'churn_rate': [churned / total_users],
        'retention_rate': [retained / total_users],
        'male_churn_rate': [male_churn],
        'female_churn_rate': [female_churn],
    })
    save(kpis, 'global_kpis.csv')
    return kpis


def generate_demographics_analytics(merged_demo):
    """Generate all demographics breakdown analytics."""
    log.info("🧑 Generating Demographics Analytics...")

    # 1. Save full demographics (for the app to use)
    cols = ['msno', 'is_churn', 'bd', 'age_group', 'gender', 'city', 'city_label',
            'registered_via', 'reg_method_label', 'reg_year', 'reg_month', 'reg_quarter', 'reg_year_month']
    cols = [c for c in cols if c in merged_demo.columns]
    save(merged_demo[cols], 'demographics_churn.parquet')

    # 2. Churn by age group
    age_churn = merged_demo.groupby('age_group', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    age_churn['churn_rate'] = age_churn['churn_count'] / age_churn['total'] * 100
    age_churn['retention_rate'] = 100 - age_churn['churn_rate']
    save(age_churn, 'age_churn.csv')

    # 3. Churn by gender
    gender_churn = merged_demo.groupby('gender', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    gender_churn['churn_rate'] = gender_churn['churn_count'] / gender_churn['total'] * 100
    save(gender_churn, 'gender_churn.csv')

    # 4. Churn by city (top 15 cities by user count)
    city_churn = merged_demo.groupby(['city', 'city_label'])['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    city_churn['churn_rate'] = city_churn['churn_count'] / city_churn['total'] * 100
    city_churn = city_churn.sort_values('total', ascending=False)
    save(city_churn, 'city_churn.csv')

    # 5. Churn by registration method
    reg_churn = merged_demo.groupby(['registered_via', 'reg_method_label'])['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    reg_churn['churn_rate'] = reg_churn['churn_count'] / reg_churn['total'] * 100
    reg_churn = reg_churn.sort_values('total', ascending=False)
    save(reg_churn, 'reg_method_churn.csv')

    # 6. Churn by registration year (cohort analysis)
    year_churn = merged_demo.groupby('reg_year')['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    year_churn['churn_rate'] = year_churn['churn_count'] / year_churn['total'] * 100
    year_churn = year_churn[year_churn['reg_year'].between(2004, 2017)]
    save(year_churn, 'cohort_year_churn.csv')

    # 7. Churn by registration month (seasonality)
    month_churn = merged_demo.groupby('reg_month')['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    month_churn['churn_rate'] = month_churn['churn_count'] / month_churn['total'] * 100
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    month_churn['month_name'] = month_churn['reg_month'].map(month_names)
    save(month_churn, 'seasonal_reg_churn.csv')

    # 8. Age vs Gender cross-tab churn rate
    age_gender_churn = merged_demo.dropna(subset=['age_group']).groupby(
        ['age_group', 'gender'], observed=False
    )['is_churn'].agg(churn_count='sum', total='count').reset_index()
    age_gender_churn['churn_rate'] = age_gender_churn['churn_count'] / age_gender_churn['total'] * 100
    save(age_gender_churn, 'age_gender_churn.csv')

    # 9. Age distribution (valid ages only)
    valid_ages = merged_demo[merged_demo['bd'].between(1, 99)][['bd', 'is_churn', 'gender']]
    save(valid_ages, 'age_distribution.parquet')

    # 10. New users per month trend
    monthly_regs = merged_demo.groupby('reg_year_month').agg(
        new_users=('msno', 'count'),
        churned=('is_churn', 'sum')
    ).reset_index()
    monthly_regs['churn_rate'] = monthly_regs['churned'] / monthly_regs['new_users'] * 100
    monthly_regs = monthly_regs[monthly_regs['reg_year_month'] != 'NaT'].sort_values('reg_year_month')
    save(monthly_regs, 'monthly_registrations.csv')


def generate_transaction_analytics(transactions, train):
    """Generate detailed transaction & revenue analytics."""
    log.info("💳 Generating Transaction Analytics...")

    # Parse dates
    trans = transactions.copy()
    trans['transaction_date'] = pd.to_datetime(
        trans['transaction_date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    trans['membership_expire_date'] = pd.to_datetime(
        trans['membership_expire_date'].astype(str), format='%Y%m%d', errors='coerce'
    )

    # Feature engineering
    trans['is_promo'] = (trans['actual_amount_paid'] == 0).astype(int)
    trans['revenue_lost'] = (trans['plan_list_price'] - trans['actual_amount_paid']).clip(lower=0)
    trans['discount_pct'] = np.where(
        trans['plan_list_price'] > 0,
        (1 - trans['actual_amount_paid'] / trans['plan_list_price']) * 100,
        0
    )
    trans['trans_year'] = trans['transaction_date'].dt.year
    trans['trans_month'] = trans['transaction_date'].dt.month
    trans['trans_year_month'] = trans['transaction_date'].dt.to_period('M').astype(str)
    trans['payment_method_label'] = trans['payment_method_id'].map(PAYMENT_METHOD_LABELS).fillna('Other')

    # Plan type classification
    def classify_plan(days):
        if days <= 7: return 'Weekly (≤7d)'
        elif days <= 31: return 'Monthly (8-31d)'
        elif days <= 100: return 'Quarterly (32-100d)'
        elif days <= 200: return 'Semi-Annual (101-200d)'
        else: return 'Annual (>200d)'
    trans['plan_type'] = trans['payment_plan_days'].apply(classify_plan)

    # Merge with churn
    trans_churn = trans.merge(train[['msno', 'is_churn']], on='msno', how='inner')

    # 1. User-level transaction KPIs
    user_trans = trans_churn.groupby('msno').agg(
        total_revenue=('actual_amount_paid', 'sum'),
        total_revenue_lost=('revenue_lost', 'sum'),
        promo_count=('is_promo', 'sum'),
        cancel_count=('is_cancel', 'sum'),
        transaction_count=('msno', 'count'),
        avg_plan_price=('plan_list_price', 'mean'),
        avg_paid=('actual_amount_paid', 'mean'),
        avg_discount_pct=('discount_pct', 'mean'),
        auto_renew_rate=('is_auto_renew', 'mean'),
        is_churn=('is_churn', 'first'),
        last_trans_date=('transaction_date', 'max'),
        first_trans_date=('transaction_date', 'min'),
    ).reset_index()
    user_trans['promo_ratio'] = user_trans['promo_count'] / user_trans['transaction_count']
    user_trans['cancel_ratio'] = user_trans['cancel_count'] / user_trans['transaction_count']
    save(user_trans, 'user_transaction_kpis.parquet')

    # 2. Revenue by plan type
    plan_rev = trans_churn.groupby(['plan_type', 'is_churn']).agg(
        total_revenue=('actual_amount_paid', 'sum'),
        user_count=('msno', 'nunique'),
        avg_revenue=('actual_amount_paid', 'mean'),
        transaction_count=('msno', 'count')
    ).reset_index()
    save(plan_rev, 'plan_type_revenue.csv')

    # 3. Monthly revenue trend
    monthly_rev = trans_churn.groupby('trans_year_month').agg(
        total_revenue=('actual_amount_paid', 'sum'),
        revenue_lost=('revenue_lost', 'sum'),
        transactions=('msno', 'count'),
        promo_count=('is_promo', 'sum')
    ).reset_index().sort_values('trans_year_month')
    monthly_rev = monthly_rev[monthly_rev['trans_year_month'] != 'NaT']
    save(monthly_rev, 'monthly_revenue.csv')

    # 4. Payment method churn
    payment_churn = trans_churn.groupby(['payment_method_label', 'is_churn']).agg(
        users=('msno', 'nunique'),
        revenue=('actual_amount_paid', 'sum'),
        transactions=('msno', 'count')
    ).reset_index()
    save(payment_churn, 'payment_method_churn.csv')

    # 5. Auto-renew impact
    auto_renew_churn = trans_churn.groupby(['is_auto_renew', 'is_churn']).agg(
        users=('msno', 'nunique'),
        revenue=('actual_amount_paid', 'sum')
    ).reset_index()
    auto_renew_churn['auto_renew_label'] = auto_renew_churn['is_auto_renew'].map({0: 'No Auto-Renew', 1: 'Auto-Renew On'})
    save(auto_renew_churn, 'auto_renew_churn.csv')

    # 6. Promo usage segmentation
    promo_bins = [-0.01, 0, 0.2, 0.5, 1.01]
    promo_labels = ['No Promos (0%)', 'Low (1-20%)', 'Medium (21-50%)', 'High (51-100%)']
    user_trans['promo_segment'] = pd.cut(user_trans['promo_ratio'], bins=promo_bins, labels=promo_labels)
    promo_seg_churn = user_trans.groupby('promo_segment', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    promo_seg_churn['churn_rate'] = promo_seg_churn['churn_count'] / promo_seg_churn['total'] * 100
    save(promo_seg_churn, 'promo_segment_churn.csv')

    # 7. Cancellation impact
    cancel_bins = [-0.01, 0, 0.1, 0.3, 1.01]
    cancel_labels = ['Never Cancelled', 'Low Cancel (<10%)', 'Medium Cancel (10-30%)', 'High Cancel (>30%)']
    user_trans['cancel_segment'] = pd.cut(user_trans['cancel_ratio'], bins=cancel_bins, labels=cancel_labels)
    cancel_seg_churn = user_trans.groupby('cancel_segment', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    cancel_seg_churn['churn_rate'] = cancel_seg_churn['churn_count'] / cancel_seg_churn['total'] * 100
    save(cancel_seg_churn, 'cancel_segment_churn.csv')

    # 8. Plan days distribution
    plan_days_churn = trans_churn.groupby(['payment_plan_days', 'is_churn']).agg(
        user_count=('msno', 'nunique')
    ).reset_index()
    # Only top 15 plan durations
    top_plans = trans_churn['payment_plan_days'].value_counts().head(15).index.tolist()
    plan_days_churn = plan_days_churn[plan_days_churn['payment_plan_days'].isin(top_plans)]
    save(plan_days_churn, 'plan_days_churn.csv')

    # 9. Revenue bucket by churn
    revenue_bins = [0, 50, 149, 300, 600, 1000, 1e9]
    revenue_labels = ['0-50', '51-149', '150-300', '301-600', '601-1000', '1000+']
    user_trans['revenue_bucket'] = pd.cut(user_trans['total_revenue'], bins=revenue_bins, labels=revenue_labels)
    rev_bucket_churn = user_trans.groupby('revenue_bucket', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    rev_bucket_churn['churn_rate'] = rev_bucket_churn['churn_count'] / rev_bucket_churn['total'] * 100
    save(rev_bucket_churn, 'revenue_bucket_churn.csv')

    log.info("  ✅ Transaction analytics done.")


def generate_engagement_analytics(train):
    """Generate engagement analytics from user logs."""
    log.info("🎧 Generating Engagement Analytics...")

    cols_logs = ['msno', 'date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']
    logs = pd.read_parquet(os.path.join(PROCESSED_DIR, 'user_logs.parquet'), columns=cols_logs)

    log.info(f"  Loaded user_logs: {logs.shape}")

    # Parse date
    logs['date'] = pd.to_datetime(logs['date'].astype(str), format='%Y%m%d', errors='coerce')
    logs['log_year'] = logs['date'].dt.year
    logs['log_month'] = logs['date'].dt.month
    logs['log_year_month'] = logs['date'].dt.to_period('M').astype(str)

    # Derived metrics
    logs['total_songs'] = logs[['num_25', 'num_50', 'num_75', 'num_985', 'num_100']].sum(axis=1)
    logs['skip_rate'] = np.where(
        logs['total_songs'] > 0,
        logs['num_25'] / logs['total_songs'],
        0
    )
    logs['completion_rate'] = np.where(
        logs['total_songs'] > 0,
        logs['num_100'] / logs['total_songs'],
        0
    )

    # 1. Per-user engagement aggregation
    user_eng = logs.groupby('msno').agg(
        active_days=('date', 'count'),
        total_secs=('total_secs', 'sum'),
        songs_played=('total_songs', 'sum'),
        unique_songs=('num_unq', 'sum'),
        songs_completed=('num_100', 'sum'),
        songs_25=('num_25', 'sum'),
        songs_50=('num_50', 'sum'),
        songs_75=('num_75', 'sum'),
        songs_985=('num_985', 'sum'),
        avg_skip_rate=('skip_rate', 'mean'),
        avg_completion_rate=('completion_rate', 'mean'),
        first_activity=('date', 'min'),
        last_activity=('date', 'max'),
    ).reset_index()
    user_eng['avg_secs_per_day'] = user_eng['total_secs'] / user_eng['active_days']
    user_eng['avg_songs_per_day'] = user_eng['songs_played'] / user_eng['active_days']
    user_eng['completion_ratio'] = np.where(
        user_eng['songs_played'] > 0,
        user_eng['songs_completed'] / user_eng['songs_played'],
        0
    )

    # Merge with churn
    user_eng = user_eng.merge(train[['msno', 'is_churn']], on='msno', how='inner')
    save(user_eng, 'user_engagement.parquet')

    # 2. Engagement segments
    secs_bins = [0, 3600, 36000, 180000, 600000, 1e12]
    secs_labels = ['<1hr', '1-10hr', '10-50hr', '50-167hr', '167hr+']
    user_eng['listening_segment'] = pd.cut(user_eng['total_secs'], bins=secs_bins, labels=secs_labels)
    listen_seg_churn = user_eng.groupby('listening_segment', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    listen_seg_churn['churn_rate'] = listen_seg_churn['churn_count'] / listen_seg_churn['total'] * 100
    save(listen_seg_churn, 'listening_segment_churn.csv')

    # 3. Active days segments
    active_bins = [-1, 5, 30, 90, 180, 10000]
    active_labels = ['1-5 days', '6-30 days', '31-90 days', '91-180 days', '180+ days']
    user_eng['active_segment'] = pd.cut(user_eng['active_days'], bins=active_bins, labels=active_labels)
    active_seg_churn = user_eng.groupby('active_segment', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    active_seg_churn['churn_rate'] = active_seg_churn['churn_count'] / active_seg_churn['total'] * 100
    save(active_seg_churn, 'active_days_segment_churn.csv')

    # 4. Skip rate segments
    skip_bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01]
    skip_labels = ['Very Low (<20%)', 'Low (20-40%)', 'Medium (40-60%)', 'High (60-80%)', 'Very High (>80%)']
    user_eng['skip_segment'] = pd.cut(user_eng['avg_skip_rate'], bins=skip_bins, labels=skip_labels)
    skip_seg_churn = user_eng.groupby('skip_segment', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    skip_seg_churn['churn_rate'] = skip_seg_churn['churn_count'] / skip_seg_churn['total'] * 100
    save(skip_seg_churn, 'skip_rate_segment_churn.csv')

    # 5. Monthly platform activity trend
    log.info("  Calculating monthly activity trend...")
    monthly_activity = logs.groupby('log_year_month').agg(
        total_secs=('total_secs', 'sum'),
        unique_users=('msno', 'nunique'),
        total_songs=('total_songs', 'sum'),
        avg_skip_rate=('skip_rate', 'mean')
    ).reset_index().sort_values('log_year_month')
    monthly_activity = monthly_activity[monthly_activity['log_year_month'] != 'NaT']
    save(monthly_activity, 'monthly_platform_activity.csv')

    # 6. Song completion breakdown by churn (avg proportions)
    eng_with_churn = user_eng[['is_churn', 'songs_25', 'songs_50', 'songs_75', 'songs_985', 'songs_completed', 'songs_played']].copy()
    pcts = eng_with_churn.groupby('is_churn').agg(
        songs_25=('songs_25', 'sum'),
        songs_50=('songs_50', 'sum'),
        songs_75=('songs_75', 'sum'),
        songs_985=('songs_985', 'sum'),
        songs_completed=('songs_completed', 'sum'),
        songs_played=('songs_played', 'sum')
    ).reset_index()
    for col in ['songs_25', 'songs_50', 'songs_75', 'songs_985', 'songs_completed']:
        pcts[col + '_pct'] = pcts[col] / pcts['songs_played'] * 100
    save(pcts, 'song_completion_breakdown.csv')

    # 7. Completion ratio bins
    compl_bins = [-0.01, 0.1, 0.3, 0.5, 0.7, 1.01]
    compl_labels = ['<10%', '10-30%', '30-50%', '50-70%', '70-100%']
    user_eng['compl_segment'] = pd.cut(user_eng['completion_ratio'], bins=compl_bins, labels=compl_labels)
    compl_seg_churn = user_eng.groupby('compl_segment', observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    compl_seg_churn['churn_rate'] = compl_seg_churn['churn_count'] / compl_seg_churn['total'] * 100
    save(compl_seg_churn, 'completion_ratio_churn.csv')

    log.info("  ✅ Engagement analytics done.")


def generate_combined_analytics(train, members):
    """Cross-feature analytics combining multiple datasets."""
    log.info("🔀 Generating Combined/Cross Analytics...")

    trans = pd.read_parquet(os.path.join(PROCESSED_DIR, 'transactions.parquet'))
    trans['transaction_date'] = pd.to_datetime(
        trans['transaction_date'].astype(str), format='%Y%m%d', errors='coerce'
    )
    trans['is_promo'] = (trans['actual_amount_paid'] == 0).astype(int)
    trans['revenue_lost'] = (trans['plan_list_price'] - trans['actual_amount_paid']).clip(lower=0)

    # User transaction aggregates
    user_trans = trans.groupby('msno').agg(
        total_revenue=('actual_amount_paid', 'sum'),
        promo_count=('is_promo', 'sum'),
        cancel_count=('is_cancel', 'sum'),
        tx_count=('msno', 'count'),
        auto_renew=('is_auto_renew', 'mean'),
    ).reset_index()
    user_trans['promo_ratio'] = user_trans['promo_count'] / user_trans['tx_count']

    # Merge engagement (drop is_churn if present to avoid collision)
    user_eng = pd.read_parquet(os.path.join(ANALYTICS_DIR, 'user_engagement.parquet'))
    if 'is_churn' in user_eng.columns:
        user_eng = user_eng.drop(columns=['is_churn'])
    user_trans_eng = user_trans.merge(user_eng, on='msno', how='inner')

    # Merge churn
    combined = user_trans_eng.merge(train[['msno', 'is_churn']], on='msno', how='inner')

    # Merge member demographics
    members_clean = members[['msno', 'bd', 'gender', 'city', 'registered_via', 'registration_init_time']].copy()
    members_clean['registration_init_time'] = pd.to_datetime(
        members_clean['registration_init_time'].astype(str), format='%Y%m%d', errors='coerce'
    )
    members_clean['reg_year'] = members_clean['registration_init_time'].dt.year
    members_clean['bd'] = members_clean['bd'].apply(lambda x: x if pd.notnull(x) and 0 < x < 100 else np.nan)
    bins = [0, 18, 25, 35, 45, 55, 100]
    labels = ['<18', '18-25', '26-35', '36-45', '46-55', '55+']
    members_clean['age_group'] = pd.cut(members_clean['bd'], bins=bins, labels=labels)

    combined = combined.merge(members_clean, on='msno', how='left')

    # 1. Revenue vs Engagement summary stats by churn
    combined_summary = combined.groupby('is_churn').agg(
        avg_revenue=('total_revenue', 'mean'),
        avg_active_days=('active_days', 'mean'),
        avg_secs=('total_secs', 'mean'),
        avg_promo_ratio=('promo_ratio', 'mean'),
        avg_cancel_count=('cancel_count', 'mean'),
        avg_auto_renew=('auto_renew', 'mean'),
        avg_completion=('avg_completion_rate', 'mean'),
        avg_skip=('avg_skip_rate', 'mean'),
        user_count=('msno', 'count')
    ).reset_index()
    save(combined_summary, 'churn_feature_summary.csv')

    # 2. Revenue tiers + active days cross
    rev_bins = [0, 50, 149, 300, 600, 1e9]
    rev_labels = ['0-50', '51-149', '150-300', '301-600', '601+']
    combined['rev_tier'] = pd.cut(combined['total_revenue'], bins=rev_bins, labels=rev_labels)
    rev_active_cross = combined.groupby(['rev_tier', 'is_churn'], observed=False)['active_days'].mean().reset_index()
    save(rev_active_cross, 'revenue_activity_cross.csv')

    # 3. Auto-renew + promo → churn funnel
    funnel = combined.groupby(['is_churn']).agg(
        auto_renew_pct=('auto_renew', 'mean'),
        promo_ratio=('promo_ratio', 'mean'),
        cancel_ratio=('cancel_count', 'mean'),
        active_days=('active_days', 'mean'),
        completion_rate=('avg_completion_rate', 'mean'),
    ).reset_index()
    save(funnel, 'churn_funnel_metrics.csv')

    # 4. Age group x payment method churn heatmap
    if 'age_group' in combined.columns and 'registered_via' in combined.columns:
        age_reg_churn = combined.dropna(subset=['age_group']).groupby(
            ['age_group', 'registered_via'], observed=False
        )['is_churn'].agg(churn_count='sum', total='count').reset_index()
        age_reg_churn['churn_rate'] = age_reg_churn['churn_count'] / age_reg_churn['total'] * 100
        age_reg_churn['reg_label'] = age_reg_churn['registered_via'].map(REG_METHOD_LABELS).fillna('Other')
        save(age_reg_churn, 'age_reg_churn_heatmap.csv')

    # 5. RFM-style segmentation
    combined['days_active_bucket'] = pd.cut(combined['active_days'],
        bins=[-1, 10, 50, 150, 10000], labels=['Rarely Active', 'Occasional', 'Regular', 'Power User'])
    combined['revenue_bucket'] = pd.cut(combined['total_revenue'],
        bins=[-1, 0, 149, 600, 1e9], labels=['No Revenue', 'Low Value', 'Mid Value', 'High Value'])

    rfm_churn = combined.groupby(['days_active_bucket', 'revenue_bucket'], observed=False)['is_churn'].agg(
        churn_count='sum', total='count'
    ).reset_index()
    rfm_churn['churn_rate'] = rfm_churn['churn_count'] / rfm_churn['total'] * 100
    save(rfm_churn, 'rfm_segment_churn.csv')

    log.info("  ✅ Combined analytics done.")


def main():
    os.makedirs(ANALYTICS_DIR, exist_ok=True)

    train, members, transactions = load_base_data()
    members_clean, merged_demo = process_members(members, train)

    generate_global_kpis(train, merged_demo)
    generate_demographics_analytics(merged_demo)
    generate_transaction_analytics(transactions, train)
    generate_engagement_analytics(train)
    generate_combined_analytics(train, members)

    log.info("🎉 ALL ANALYTICS GENERATED SUCCESSFULLY!")
    log.info(f"  Files saved to: {ANALYTICS_DIR}")


if __name__ == "__main__":
    main()
