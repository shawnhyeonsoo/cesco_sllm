"""
CESCO Customer Churn Dataset Builder
=====================================

This script builds the complete ML dataset for customer churn prediction.
It performs the following steps:
1. Load data from RODB and BIDB databases
2. Create churn labels based on contract termination logic
3. Extract features from JSON columns (contracts, interactions, work history, purchases)
   - For churned customers: uses data up to ONE MONTH PRIOR to churn date
   - For non-churned customers: uses data up to ONE MONTH PRIOR to reference date (2025-10-30)
4. Create derived features (lifecycle, engagement, risk scores)
5. Create time-windowed features (30/60/90/180/365 days)
6. Save the final dataset

Author: Generated for CESCO ML Project
Date: 2025-10-30
Version: 2.0 - Added temporal cutoff logic for prediction modeling
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import custom database connection classes
from src.data.dataloader import CescoRodbConnection, CescoBidbConnection


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_SIZE = None  # Number of customers to sample (set to None for all)
OUTPUT_DIR = "data/processed"
RANDOM_STATE = 42
REFERENCE_DATE = datetime(2025, 10, 30)  # Today's date for non-churned customers

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Database Queries (Loaded from SQL files)
# ============================================================================
# RODB query: ./assets/queries/rodb_query_train.sql
# BIDB query: ./assets/queries/bidb_query_custcode_with_mall_data.sql


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_contract_features(contracts_json, reference_date=None):
    """Extract 14 contract-related features based on data up to reference_date"""
    if reference_date is None:
        reference_date = datetime.now()
    
    if pd.isna(contracts_json) or not contracts_json:
        return {
            'contract_count': 0, 'contract_active_count': 0, 'contract_cancelled_count': 0,
            'contract_days_since_first': np.nan, 'contract_days_since_last': np.nan,
            'contract_avg_duration': np.nan, 'contract_total_duration': 0,
            'contract_cancellation_rate': 0, 'contract_most_common_cycle': 'unknown',
            'contract_most_common_service': 'unknown', 'contract_unique_products': 0,
            'contract_has_active': 0, 'contract_recent_cancellation': 0,
            'contract_lifecycle_stage': 'unknown'
        }
    
    try:
        contracts = json.loads(contracts_json)
        if not contracts:
            return extract_contract_features(None, reference_date)
        
        # Filter contracts that started before reference_date
        valid_contracts = []
        for c in contracts:
            start_str = c.get('시작일자')  # Changed from '신계약일자' to '시작일자'
            if start_str:
                try:
                    start = datetime.strptime(str(start_str), '%Y%m%d')
                    if start <= reference_date:
                        valid_contracts.append(c)
                except:
                    pass
        
        if not valid_contracts:
            return extract_contract_features(None, reference_date)
        
        total = len(valid_contracts)
        active = sum(1 for c in valid_contracts if c.get('해약여부') == 'N')
        cancelled = total - active
        
        dates = []
        durations = []
        for c in valid_contracts:
            start_str = c.get('시작일자')  # Changed from '신계약일자' to '시작일자'
            end_str = c.get('해약일자')
            
            if start_str:
                try:
                    start = datetime.strptime(str(start_str), '%Y%m%d')
                    dates.append(start)
                    
                    if end_str:
                        end = datetime.strptime(str(end_str), '%Y%m%d')
                        # Only count cancellations that happened before reference_date
                        if end <= reference_date:
                            durations.append((end - start).days)
                except:
                    pass
        
        days_since_first = (reference_date - min(dates)).days if dates else np.nan
        days_since_last = (reference_date - max(dates)).days if dates else np.nan
        avg_duration = np.mean(durations) if durations else np.nan
        total_duration = sum(durations) if durations else 0
        
        cancellation_rate = cancelled / total if total > 0 else 0
        
        # Use actual fields from the data: 계약유형 (contract type) instead of non-existent 서비스주기코드
        cycles = [c.get('계약유형') for c in valid_contracts if c.get('계약유형')]
        most_common_cycle = max(set(cycles), key=cycles.count) if cycles else 'unknown'
        
        # Use 계약대상_중분류명 (contract target category) instead of non-existent 서비스구분코드
        services = [c.get('계약대상_중분류명') for c in valid_contracts if c.get('계약대상_중분류명')]
        most_common_service = max(set(services), key=services.count) if services else 'unknown'
        
        # Use 계약대상 (contract target) as a proxy for product since 상품코드 doesn't exist
        unique_products = len(set(c.get('계약대상') for c in valid_contracts if c.get('계약대상')))
        
        has_active = 1 if active > 0 else 0
        recent_cancellation = 1 if cancelled > 0 and days_since_last < 180 else 0
        
        if days_since_first < 365:
            lifecycle = 'new'
        elif days_since_first < 1095:
            lifecycle = 'mature'
        else:
            lifecycle = 'veteran'
        
        return {
            'contract_count': total,
            'contract_active_count': active,
            'contract_cancelled_count': cancelled,
            'contract_days_since_first': days_since_first,
            'contract_days_since_last': days_since_last,
            'contract_avg_duration': avg_duration,
            'contract_total_duration': total_duration,
            'contract_cancellation_rate': cancellation_rate,
            'contract_most_common_cycle': most_common_cycle,
            'contract_most_common_service': most_common_service,
            'contract_unique_products': unique_products,
            'contract_has_active': has_active,
            'contract_recent_cancellation': recent_cancellation,
            'contract_lifecycle_stage': lifecycle
        }
    except:
        return extract_contract_features(None, reference_date)


def extract_interaction_features_fixed(interaction_json, reference_date=None):
    """Extract 9 interaction-related features (FIXED version using Korean field names)"""
    if reference_date is None:
        reference_date = datetime.now()
    
    if pd.isna(interaction_json) or not interaction_json:
        return {
            'interaction_count': 0, 'interaction_days_since_last': np.nan,
            'interaction_days_since_first': np.nan, 'interaction_avg_frequency': 0,
            'interaction_most_common_channel': 'unknown', 'interaction_unique_channels': 0,
            'interaction_most_common_type': 'unknown', 'interaction_complaint_ratio': 0,
            'interaction_recent_activity': 0
        }
    
    try:
        interactions = json.loads(interaction_json)
        if not interactions:
            return extract_interaction_features_fixed(None, reference_date)
        
        # Filter interactions that occurred before reference_date
        valid_interactions = []
        dates = []
        for i in interactions:
            date_str = i.get('접수일시')  # Changed from English to Korean
            if date_str:
                try:
                    date = pd.to_datetime(date_str)
                    if date <= reference_date:
                        valid_interactions.append(i)
                        dates.append(date)
                except:
                    pass
        
        if not valid_interactions:
            return extract_interaction_features_fixed(None, reference_date)
        
        total = len(valid_interactions)
        days_since_last = (reference_date - max(dates)).days if dates else np.nan
        days_since_first = (reference_date - min(dates)).days if dates else np.nan
        
        avg_frequency = total / (days_since_first / 30) if days_since_first and days_since_first > 0 else 0
        
        channels = [i.get('접수채널구분명') for i in valid_interactions if i.get('접수채널구분명')]
        most_common_channel = max(set(channels), key=channels.count) if channels else 'unknown'
        unique_channels = len(set(channels)) if channels else 0
        
        types = [i.get('문의유형대분류명') for i in valid_interactions if i.get('문의유형대분류명')]
        most_common_type = max(set(types), key=types.count) if types else 'unknown'
        
        complaints = sum(1 for i in valid_interactions if '불만' in str(i.get('문의유형대분류명', '')))
        complaint_ratio = complaints / total if total > 0 else 0
        
        recent_activity = 1 if days_since_last and days_since_last < 30 else 0
        
        return {
            'interaction_count': total,
            'interaction_days_since_last': days_since_last,
            'interaction_days_since_first': days_since_first,
            'interaction_avg_frequency': avg_frequency,
            'interaction_most_common_channel': most_common_channel,
            'interaction_unique_channels': unique_channels,
            'interaction_most_common_type': most_common_type,
            'interaction_complaint_ratio': complaint_ratio,
            'interaction_recent_activity': recent_activity
        }
    except:
        return extract_interaction_features_fixed(None, reference_date)


def extract_work_features(work_json, reference_date=None):
    """Extract 12 work-related features"""
    if reference_date is None:
        reference_date = datetime.now()
    
    if pd.isna(work_json) or not work_json:
        return {
            'work_count': 0, 'work_days_since_last': np.nan, 'work_days_since_first': np.nan,
            'work_avg_frequency': 0, 'work_completion_rate': 0, 'work_avg_csi': np.nan,
            'work_csi_std': np.nan, 'work_recent_activity': 0, 'work_most_common_type': 'unknown',
            'work_unique_types': 0, 'work_incomplete_count': 0, 'work_low_csi_ratio': 0
        }
    
    try:
        works = json.loads(work_json)
        if not works:
            return extract_work_features(None, reference_date)
        
        # Filter works that occurred before reference_date
        valid_works = []
        dates = []
        for w in works:
            date_str = w.get('작업일자')
            if date_str:
                try:
                    if len(str(date_str)) == 8:
                        date = datetime.strptime(str(date_str), '%Y%m%d')
                        if date <= reference_date:
                            valid_works.append(w)
                            dates.append(date)
                except:
                    pass
        
        if not valid_works:
            return extract_work_features(None, reference_date)
        
        total = len(valid_works)
        days_since_last = (reference_date - max(dates)).days if dates else np.nan
        days_since_first = (reference_date - min(dates)).days if dates else np.nan
        
        avg_frequency = total / (days_since_first / 30) if days_since_first and days_since_first > 0 else 0
        
        # Use 확정여부 (confirmation status) instead of non-existent 완료여부
        completed = sum(1 for w in valid_works if w.get('확정여부') == True or w.get('확정여부') == 'True' or w.get('확정여부') == 'Y')
        completion_rate = completed / total if total > 0 else 0
        
        # Extract CSI scores from nested 서비스_만족도 (service satisfaction) JSON
        csi_scores = []
        for w in valid_works:
            satisfaction = w.get('서비스_만족도')
            if satisfaction:
                try:
                    if isinstance(satisfaction, str):
                        sat_data = json.loads(satisfaction)
                        if sat_data and len(sat_data) > 0:
                            for s in sat_data:
                                csi = s.get('평균_CSI_점수')
                                if csi is not None:
                                    csi_scores.append(float(csi))
                except:
                    pass
        
        avg_csi = np.mean(csi_scores) if csi_scores else np.nan
        csi_std = np.std(csi_scores) if len(csi_scores) > 1 else np.nan
        
        recent_activity = 1 if days_since_last and days_since_last < 30 else 0
        
        # Use 작업유형 (work type) instead of non-existent 작업구분코드
        types = [w.get('작업유형') for w in valid_works if w.get('작업유형')]
        most_common_type = max(set(types), key=types.count) if types else 'unknown'
        unique_types = len(set(types)) if types else 0
        
        incomplete_count = total - completed
        low_csi = sum(1 for score in csi_scores if score < 70)
        low_csi_ratio = low_csi / len(csi_scores) if csi_scores else 0
        
        return {
            'work_count': total,
            'work_days_since_last': days_since_last,
            'work_days_since_first': days_since_first,
            'work_avg_frequency': avg_frequency,
            'work_completion_rate': completion_rate,
            'work_avg_csi': avg_csi,
            'work_csi_std': csi_std,
            'work_recent_activity': recent_activity,
            'work_most_common_type': most_common_type,
            'work_unique_types': unique_types,
            'work_incomplete_count': incomplete_count,
            'work_low_csi_ratio': low_csi_ratio
        }
    except:
        return extract_work_features(None, reference_date)


def extract_purchase_features(purchase_json, reference_date=None):
    """Extract 15 purchase-related features"""
    import pandas as pd
    import json
    from datetime import datetime
    import numpy as np
    
    if reference_date is None:
        reference_date = datetime.now()
    
    if pd.isna(purchase_json) or not purchase_json:
        return {
            'purchase_count': 0, 'purchase_total_amount': 0, 'purchase_avg_amount': 0,
            'purchase_total_items': 0, 'purchase_avg_items': 0, 'purchase_days_since_last': np.nan,
            'purchase_days_since_first': np.nan, 'purchase_frequency': 0,
            'purchase_unique_products': 0, 'purchase_most_common_product': 'none',
            'purchase_recent_30d': 0, 'purchase_recent_90d': 0, 'purchase_recent_180d': 0,
            'purchase_amount_std': 0, 'purchase_items_std': 0
        }
    
    try:
        purchases = json.loads(purchase_json)
        if not purchases:
            return extract_purchase_features(None, reference_date)
        
        # Filter purchases that occurred before reference_date
        valid_purchases = []
        dates = []
        for p in purchases:
            date_str = p.get('purchase_date')
            if date_str:
                try:
                    if len(str(date_str)) == 8:
                        date = datetime.strptime(str(date_str), '%Y%m%d')
                        if date <= reference_date:
                            valid_purchases.append(p)
                            dates.append(date)
                except:
                    pass
        
        if not valid_purchases:
            return extract_purchase_features(None, reference_date)
        
        total = len(valid_purchases)
        
        amounts = [p.get('판매금액', 0) for p in valid_purchases]
        items = [p.get('판매수량', 0) for p in valid_purchases]
        
        total_amount = sum(amounts)
        avg_amount = np.mean(amounts) if amounts else 0
        total_items = sum(items)
        avg_items = np.mean(items) if items else 0
        
        days_since_last = (reference_date - max(dates)).days if dates else np.nan
        days_since_first = (reference_date - min(dates)).days if dates else np.nan
        
        frequency = total / (days_since_first / 30) if days_since_first and days_since_first > 0 else 0
        
        products = [p.get('상품종류명') for p in valid_purchases if p.get('상품종류명')]
        unique_products = len(set(products)) if products else 0
        most_common_product = max(set(products), key=products.count) if products else 'none'
        
        recent_30d = sum(1 for d in dates if (reference_date - d).days <= 30)
        recent_90d = sum(1 for d in dates if (reference_date - d).days <= 90)
        recent_180d = sum(1 for d in dates if (reference_date - d).days <= 180)
        
        amount_std = np.std(amounts) if len(amounts) > 1 else 0
        items_std = np.std(items) if len(items) > 1 else 0
        
        return {
            'purchase_count': total,
            'purchase_total_amount': total_amount,
            'purchase_avg_amount': avg_amount,
            'purchase_total_items': total_items,
            'purchase_avg_items': avg_items,
            'purchase_days_since_last': days_since_last,
            'purchase_days_since_first': days_since_first,
            'purchase_frequency': frequency,
            'purchase_unique_products': unique_products,
            'purchase_most_common_product': most_common_product,
            'purchase_recent_30d': recent_30d,
            'purchase_recent_90d': recent_90d,
            'purchase_recent_180d': recent_180d,
            'purchase_amount_std': amount_std,
            'purchase_items_std': items_std
        }
    except:
        return extract_purchase_features(None, reference_date)


def extract_time_windowed_features(row, reference_date=None):
    """Extract time-windowed features (30/60/90/180/365 days) based on reference_date"""
    if reference_date is None:
        reference_date = datetime.now()
    
    features = {}
    
    # Interaction time windows
    try:
        if pd.notna(row['interaction_history']):
            interactions = json.loads(row['interaction_history'])
            dates = []
            for i in interactions:
                date_str = i.get('접수일시')
                if date_str:
                    try:
                        date = pd.to_datetime(date_str)
                        if date <= reference_date:
                            dates.append(date)
                    except:
                        pass
            
            for window in [30, 60, 90, 180, 365]:
                cutoff = reference_date - timedelta(days=window)
                features[f'interaction_last_{window}_days'] = sum(1 for d in dates if d >= cutoff)
            
            # Ratios
            total = len(dates) if dates else 1
            features['interaction_ratio_30_to_total'] = features.get('interaction_last_30_days', 0) / total
            features['interaction_ratio_90_to_total'] = features.get('interaction_last_90_days', 0) / total
            
            # Trend (30d vs 90d)
            features['interaction_trend_30_vs_90'] = (
                features.get('interaction_last_30_days', 0) - 
                features.get('interaction_last_90_days', 0) / 3
            ) if features.get('interaction_last_90_days', 0) > 0 else 0
            
            # Decay score
            decay_sum = sum(np.exp(-((reference_date - d).days / 180)) for d in dates)
            features['interaction_decay_score'] = decay_sum
        else:
            for window in [30, 60, 90, 180, 365]:
                features[f'interaction_last_{window}_days'] = 0
            features['interaction_ratio_30_to_total'] = 0
            features['interaction_ratio_90_to_total'] = 0
            features['interaction_trend_30_vs_90'] = 0
            features['interaction_decay_score'] = 0
    except:
        for window in [30, 60, 90, 180, 365]:
            features[f'interaction_last_{window}_days'] = 0
        features['interaction_ratio_30_to_total'] = 0
        features['interaction_ratio_90_to_total'] = 0
        features['interaction_trend_30_vs_90'] = 0
        features['interaction_decay_score'] = 0
    
    # Work time windows
    try:
        if pd.notna(row['작업이력']):
            works = json.loads(row['작업이력'])
            dates = []
            for w in works:
                date_str = w.get('작업일자')
                if date_str and len(str(date_str)) == 8:
                    try:
                        date = datetime.strptime(str(date_str), '%Y%m%d')
                        if date <= reference_date:
                            dates.append(date)
                    except:
                        pass
            
            for window in [30, 60, 90, 180, 365]:
                cutoff = reference_date - timedelta(days=window)
                features[f'work_last_{window}_days'] = sum(1 for d in dates if d >= cutoff)
            
            # Ratios
            total = len(dates) if dates else 1
            features['work_ratio_30_to_total'] = features.get('work_last_30_days', 0) / total
            features['work_ratio_90_to_total'] = features.get('work_last_90_days', 0) / total
            
            # Trend
            features['work_trend_30_vs_90'] = (
                features.get('work_last_30_days', 0) - 
                features.get('work_last_90_days', 0) / 3
            ) if features.get('work_last_90_days', 0) > 0 else 0
            
            # Decay score
            decay_sum = sum(np.exp(-((reference_date - d).days / 180)) for d in dates)
            features['work_decay_score'] = decay_sum
        else:
            for window in [30, 60, 90, 180, 365]:
                features[f'work_last_{window}_days'] = 0
            features['work_ratio_30_to_total'] = 0
            features['work_ratio_90_to_total'] = 0
            features['work_trend_30_vs_90'] = 0
            features['work_decay_score'] = 0
    except:
        for window in [30, 60, 90, 180, 365]:
            features[f'work_last_{window}_days'] = 0
        features['work_ratio_30_to_total'] = 0
        features['work_ratio_90_to_total'] = 0
        features['work_trend_30_vs_90'] = 0
        features['work_decay_score'] = 0
    
    return pd.Series(features)


def create_derived_features(df):
    """Create 36 advanced derived features"""
    derived = pd.DataFrame(index=df.index)
    
    # Contract lifecycle features (6)
    derived['contract_age_years'] = df['contract_days_since_first'] / 365
    derived['contract_avg_duration_years'] = df['contract_avg_duration'] / 365
    derived['contract_churn_risk'] = df['contract_cancellation_rate'] * (1 + df['contract_recent_cancellation'])
    derived['contract_stability'] = df['contract_active_count'] / (df['contract_count'] + 1)
    derived['contract_diversity'] = df['contract_unique_products'] / (df['contract_count'] + 1)
    derived['contract_renewal_indicator'] = (df['contract_count'] > 1).astype(int)
    
    # Interaction engagement features (8)
    derived['interaction_engagement_score'] = (
        df['interaction_avg_frequency'] * 
        (1 - df['interaction_complaint_ratio']) * 
        df['interaction_unique_channels']
    )
    derived['interaction_recency_score'] = 1 / (df['interaction_days_since_last'] + 1)
    derived['interaction_history_years'] = df['interaction_days_since_first'] / 365
    derived['interaction_per_contract'] = df['interaction_count'] / (df['contract_count'] + 1)
    derived['interaction_complaint_absolute'] = df['interaction_count'] * df['interaction_complaint_ratio']
    derived['interaction_intensity'] = df['interaction_count'] / (df['interaction_days_since_first'] + 1)
    derived['interaction_channel_diversity'] = df['interaction_unique_channels'] / (df['interaction_count'] + 1)
    derived['interaction_recent_surge'] = (df['interaction_recent_activity'] * df['interaction_avg_frequency']).clip(upper=10)
    
    # Work quality features (8)
    derived['work_quality_score'] = (
        df['work_completion_rate'] * 
        (df['work_avg_csi'] / 100) * 
        (1 - df['work_low_csi_ratio'])
    )
    derived['work_reliability'] = df['work_completion_rate'] * (1 - df['work_incomplete_count'] / (df['work_count'] + 1))
    derived['work_recency_score'] = 1 / (df['work_days_since_last'] + 1)
    derived['work_history_years'] = df['work_days_since_first'] / 365
    derived['work_per_contract'] = df['work_count'] / (df['contract_count'] + 1)
    derived['work_csi_trend'] = df['work_avg_csi'] - 80  # Deviation from expected 80
    derived['work_consistency'] = 1 / (df['work_csi_std'] + 1)
    derived['work_diversity'] = df['work_unique_types'] / (df['work_count'] + 1)
    
    # Combined recency (3)
    derived['recency_combined'] = (
        derived['contract_age_years'].fillna(0) +
        derived['interaction_history_years'].fillna(0) +
        derived['work_history_years'].fillna(0)
    ) / 3
    derived['recently_active_interaction'] = (df['interaction_days_since_last'] < 30).astype(int)
    derived['recently_active_work'] = (df['work_days_since_last'] < 30).astype(int)
    
    # Risk indicators (4)
    derived['service_degradation_risk'] = (
        (1 - df['work_completion_rate']) * 
        df['work_low_csi_ratio'] * 
        (df['work_days_since_last'] < 90).astype(int)
    )
    derived['complaint_risk'] = (
        df['interaction_complaint_ratio'] * 
        df['interaction_avg_frequency']
    )
    derived['is_dormant'] = ((df['interaction_days_since_last'] > 180) & (df['work_days_since_last'] > 180)).astype(int)
    derived['pest_problem_indicator'] = (df['work_avg_csi'] < 70).astype(int)
    
    # Purchase behavior features (8)
    derived['purchase_engagement'] = df['purchase_frequency'] * df['purchase_avg_amount']
    derived['purchase_recency_score'] = 1 / (df['purchase_days_since_last'] + 1)
    derived['purchase_diversity'] = df['purchase_unique_products'] / (df['purchase_count'] + 1)
    derived['purchase_loyalty'] = (df['purchase_count'] > 3).astype(int)
    derived['purchase_trend_30_to_180'] = (
        df['purchase_recent_30d'] - df['purchase_recent_180d'] / 6
    )
    derived['purchase_consistency'] = 1 / (df['purchase_amount_std'] + 1)
    derived['purchase_per_contract'] = df['purchase_count'] / (df['contract_count'] + 1)
    derived['has_purchases'] = (df['purchase_count'] > 0).astype(int)
    
    return derived


def create_churn_label(df):
    """
    Create churn label using logic from 해약여부관련_정리_v0_2.ipynb
    A customer is churned (해약여부=1) if ALL contracts have 해약일자
    A customer is retained (해약여부=0) if at least one contract has NO 해약일자
    Also adds cancellation dates (최근_해약일자_SM, 최초_해약일자_SM)
    """
    labels = []
    최근_해약일자_list = []
    최초_해약일자_list = []
    
    for idx, row in df.iterrows():
        해약여부 = 0  # Default: not churned (retained)
        최근_해약일자 = None
        최초_해약일자 = None
        
        # Parse contracts
        try:
            if pd.notna(row['contracts_info']):
                contracts = json.loads(row['contracts_info'])
                
                if contracts and len(contracts) > 0:
                    # Count contracts with and without cancellation dates
                    contracts_with_해약일자 = sum(1 for c in contracts if pd.notna(c.get('해약일자')) and c.get('해약일자') != '')
                    contracts_without_해약일자 = sum(1 for c in contracts if pd.isna(c.get('해약일자')) or c.get('해약일자') == '')
                    
                    # Customer is churned ONLY if ALL contracts have 해약일자
                    # (all contracts are cancelled)
                    if contracts_without_해약일자 == 0 and contracts_with_해약일자 > 0:
                        해약여부 = 1
                    else:
                        해약여부 = 0  # At least one active contract
                    
                    # Extract cancellation dates
                    해약일자_list = [c.get('해약일자') for c in contracts if c.get('해약일자') and c.get('해약일자') != '']
                    if 해약일자_list:
                        try:
                            dates = [datetime.strptime(str(d), '%Y%m%d') for d in 해약일자_list]
                            최근_해약일자 = max(dates)
                            최초_해약일자 = min(dates)
                        except Exception:
                            pass
        except Exception:
            pass
        
        labels.append(해약여부)
        최근_해약일자_list.append(최근_해약일자)
        최초_해약일자_list.append(최초_해약일자)
    
    df['해약여부'] = labels
    df['최근_해약일자_SM'] = 최근_해약일자_list
    df['최초_해약일자_SM'] = 최초_해약일자_list
    
    return df


def encode_categorical_features(df):
    """Encode categorical features using LabelEncoder"""
    categorical_cols = [
        'contract_most_common_cycle', 'contract_most_common_service',
        'contract_lifecycle_stage', 'interaction_most_common_channel',
        'interaction_most_common_type', 'work_most_common_type',
        'purchase_most_common_product'
    ]
    
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    return df


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("=" * 80)
    print("CESCO CUSTOMER CHURN DATASET BUILDER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'ALL'}")
    print()
    
    # ========================================================================
    # Step 1: Connect to databases
    # ========================================================================
    print("Step 1: Connecting to databases...")
    try:
        rodb = CescoRodbConnection()
        if not rodb.connect():
            print("   ✗ RODB connection failed")
            return
        print("   ✓ Connected to RODB (CESCOEIS)")
        
        bidb = CescoBidbConnection()
        if not bidb.connect():
            print("   ✗ BIDB connection failed")
            return
        print("   ✓ Connected to BIDB (CX_CDM)")
    except Exception as e:
        print(f"   ✗ Database connection failed: {e}")
        return
    
    # ========================================================================
    # Step 2: Load customer data from RODB
    # ========================================================================
    print("\nStep 2: Loading customer data from RODB...")
    
    # Load query from file
    with open("./assets/queries/rodb_query_train.sql", "r", encoding="utf-8") as file:
        rodb_query = file.read()
    
    rodb_data = rodb.execute_query(rodb_query)
    
    if rodb_data is None or len(rodb_data) == 0:
        print("   ✗ No data returned from RODB")
        return
    
    # Limit to sample size
    if SAMPLE_SIZE and len(rodb_data) > SAMPLE_SIZE:
        rodb_data = rodb_data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    print(f"   ✓ Loaded RODB data: {rodb_data.shape}")
    
    # Get customer codes for BIDB query
    custcodes = rodb_data['고객코드'].unique().tolist()
    print(f"   ✓ Unique customers: {len(custcodes)}")
    
    # ========================================================================
    # Step 3: Load work history and purchase data from BIDB
    # ========================================================================
    print("\nStep 3: Loading work history and purchase data from BIDB...")
    
    # Load query from file and format with customer codes
    custcode_list = ', '.join(f"'{code}'" for code in custcodes)
    with open("./assets/queries/bidb_query_custcode_with_mall_data copy.sql", "r", encoding="utf-8") as file:
        bidb_query_template = file.read()
    bidb_query = bidb_query_template.format(custcode_list=custcode_list)
    
    bidb_data = bidb.execute_query(bidb_query)
    print(f"   ✓ Loaded BIDB data: {bidb_data.shape}")

    ### Step 3.1: Get MyLab Data from BIDB
    print("\nStep 3.1: Loading MyLab data from BIDB...")
    with open("./assets/queries/mylab_query.sql", "r", encoding="utf-8") as file:
        mylab_query = file.read()
    mylab_query_formatted = mylab_query.format(custcode_list=custcode_list)

    print(f"Prepared MyLab query: {mylab_query_formatted[:100]}...")  # Print first 100 chars
    mylab_df = bidb.execute_query(mylab_query_formatted)

    bidb_data = bidb_data.merge(mylab_df, on='고객코드', how='left')

    if bidb_data is None or len(bidb_data) == 0:
        print("   ⚠️ No BIDB data found, continuing without work/purchase features")
        final_df = rodb_data.copy()
        final_df['purchase_logs'] = None
        final_df['작업이력'] = None
    else:
        print(f"   ✓ Loaded BIDB data: {bidb_data.shape}")
        # Merge BIDB data (contains 작업이력 and purchase_logs)
        final_df = rodb_data.merge(bidb_data, on='고객코드', how='left')
        print(f"   ✓ Merged data: {final_df.shape}")
        print(f"   ✓ Columns: {', '.join(final_df.columns.tolist())}")
    



    # ========================================================================
    # Step 4: Create churn labels
    # ========================================================================
    print("\nStep 4: Creating churn labels...")
    final_df = create_churn_label(final_df)
    churn_rate = final_df['해약여부'].mean()
    churn_count = final_df['해약여부'].sum()
    print(f"   ✓ Churn rate: {churn_rate*100:.2f}% ({churn_count} churned)")
    
    # ========================================================================
    # Step 5: Calculate reference dates and extract features from JSON columns
    # ========================================================================
    print("\nStep 5: Calculating reference dates and extracting features...")
    
    # Calculate reference date for each customer
    # - Churned customers: 1 month before churn date
    # - Non-churned customers: 1 month before today (REFERENCE_DATE)
    def get_reference_date(row):
        if row['해약여부'] == 1 and pd.notna(row['최근_해약일자_SM']):
            # Churned: use 1 month before churn date
            churn_date = row['최근_해약일자_SM']
            ref_date = churn_date - timedelta(days=30)
        else:
            # Not churned: use 1 month before today
            ref_date = REFERENCE_DATE - timedelta(days=30)
        return ref_date
    
    final_df['reference_date'] = final_df.apply(get_reference_date, axis=1)
    print(f"   ✓ Reference dates calculated")
    print(f"     - Churned customers: 1 month before churn date")
    print(f"     - Non-churned customers: 1 month before {REFERENCE_DATE.strftime('%Y-%m-%d')}")
    
    # Contract features (14)
    print("   - Extracting contract features...")
    contract_features = final_df.apply(
        lambda row: extract_contract_features(row['contracts_info'], row['reference_date']), 
        axis=1
    )
    contract_df = pd.DataFrame(contract_features.tolist())
    
    # Interaction features (9)
    print("   - Extracting interaction features...")
    interaction_features = final_df.apply(
        lambda row: extract_interaction_features_fixed(row['interaction_history'], row['reference_date']), 
        axis=1
    )
    interaction_df = pd.DataFrame(interaction_features.tolist())
    
    # Work features (12)
    print("   - Extracting work features...")
    work_features = final_df.apply(
        lambda row: extract_work_features(row['작업이력'], row['reference_date']), 
        axis=1
    )
    work_df = pd.DataFrame(work_features.tolist())
    
    # Purchase features (15)
    print("   - Extracting purchase features...")
    purchase_features = final_df.apply(
        lambda row: extract_purchase_features(row['purchase_logs'], row['reference_date']), 
        axis=1
    )
    purchase_df = pd.DataFrame(purchase_features.tolist())
    
    # Combine all features
    ml_dataset = pd.concat([
        final_df[['고객코드', '고객명', '해약여부', '최근_해약일자_SM', '최초_해약일자_SM']],
        contract_df,
        interaction_df,
        work_df,
        purchase_df
    ], axis=1)
    
    print(f"   ✓ Base features extracted: {ml_dataset.shape[1] - 5} features")
    
    # ========================================================================
    # Step 6: Create time-windowed features
    # ========================================================================
    print("\nStep 6: Creating time-windowed features...")
    time_features = final_df.apply(
        lambda row: extract_time_windowed_features(row, row['reference_date']), 
        axis=1
    )
    ml_dataset = pd.concat([ml_dataset, time_features], axis=1)
    print(f"   ✓ Time-windowed features added: 20 features")
    print(f"   ✓ Current total: {ml_dataset.shape[1] - 5} features")
    
    # ========================================================================
    # Step 7: Create derived features
    # ========================================================================
    print("\nStep 7: Creating derived features...")
    derived_features = create_derived_features(ml_dataset)
    ml_dataset = pd.concat([ml_dataset, derived_features], axis=1)
    print(f"   ✓ Derived features added: {derived_features.shape[1]} features")
    print(f"   ✓ Current total: {ml_dataset.shape[1] - 5} features")
    
    # ========================================================================
    # Step 8: Encode categorical features
    # ========================================================================
    print("\nStep 8: Encoding categorical features...")
    ml_dataset = encode_categorical_features(ml_dataset)
    print(f"   ✓ Categorical features encoded")
    print(f"   ✓ Final total: {ml_dataset.shape[1] - 5} features")
    
    # ========================================================================
    # Step 9: Save dataset
    # ========================================================================
    print("\nStep 9: Saving dataset...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save with timestamp
    csv_file = f"{OUTPUT_DIR}/ml_dataset_WITH_PURCHASE_{timestamp}.csv"
    pkl_file = f"{OUTPUT_DIR}/ml_dataset_WITH_PURCHASE_{timestamp}.pkl"
    
    ml_dataset.to_csv(csv_file, index=False, encoding='utf-8-sig')
    ml_dataset.to_pickle(pkl_file)
    
    print(f"   ✓ CSV saved: {csv_file}")
    print(f"     Size: {os.path.getsize(csv_file) / 1024 / 1024:.2f} MB")
    print(f"   ✓ Pickle saved: {pkl_file}")
    print(f"     Size: {os.path.getsize(pkl_file) / 1024 / 1024:.2f} MB")
    
    # Also save as "latest"
    latest_csv = f"{OUTPUT_DIR}/ml_dataset_pre_final_latest.csv"
    latest_pkl = f"{OUTPUT_DIR}/ml_dataset_pre_final_latest.pkl"
    
    ml_dataset.to_csv(latest_csv, index=False, encoding='utf-8-sig')
    ml_dataset.to_pickle(latest_pkl)
    print(f"   ✓ Also saved as 'latest' versions")
    
    # Save feature list
    feature_cols = [col for col in ml_dataset.columns 
                   if col not in ['고객코드', '고객명', '해약여부', '최근_해약일자_SM', '최초_해약일자_SM']]
    with open(f"{OUTPUT_DIR}/feature_list_WITH_PURCHASE_{timestamp}.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"   ✓ Feature list saved ({len(feature_cols)} features)")
    
    # ========================================================================
    # Step 10: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("DATASET BUILD COMPLETE!")
    print("=" * 80)
    print(f"Total samples: {len(ml_dataset):,}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Dataset shape: {ml_dataset.shape}")
    print(f"\nChurn Statistics:")
    print(f"  - Churn rate: {ml_dataset['해약여부'].mean()*100:.2f}%")
    print(f"  - Churned customers: {ml_dataset['해약여부'].sum()}")
    print(f"  - Retained customers: {(1-ml_dataset['해약여부']).sum()}")
    print(f"  - Imbalance ratio: 1:{(1-ml_dataset['해약여부']).sum() / ml_dataset['해약여부'].sum():.1f}")
    
    print(f"\nPurchase Data Coverage:")
    has_purchases = (ml_dataset['purchase_count'] > 0).sum()
    print(f"  - Customers with purchases: {has_purchases} / {len(ml_dataset)} ({has_purchases/len(ml_dataset)*100:.1f}%)")
    
    print(f"\nOutput Files:")
    print(f"  - Main dataset: {latest_pkl}")
    print(f"  - CSV version: {latest_csv}")
    print(f"  - Feature list: feature_list_WITH_PURCHASE_{timestamp}.pkl")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Display column verification for testing
    print("\n" + "=" * 80)
    print("COLUMN VERIFICATION")
    print("=" * 80)
    expected_json_cols = ['contracts_info', 'interaction_history', '작업이력', 'purchase_logs']
    print(f"\n✅ Expected JSON columns:")
    for col in expected_json_cols:
        has_col = col in final_df.columns
        status = "✓" if has_col else "✗"
        print(f"   {status} {col}")
    
    feature_cols = [col for col in ml_dataset.columns 
                   if col not in ['고객코드', '고객명', '해약여부', '최근_해약일자_SM', '최초_해약일자_SM']]
    print(f"\n✅ Final Feature Count: {len(feature_cols)}")
    print(f"✅ Sample Features: {', '.join(feature_cols[:10])}...")
    print("=" * 80)
    
    return ml_dataset


if __name__ == "__main__":
    ml_dataset = main()
