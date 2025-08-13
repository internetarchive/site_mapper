import pandas as pd
import numpy as np  
from urllib.parse import urlparse
import re
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def extract_url_features(url):
    """Extract meaningful features from URL instead of using raw URL"""
    try:
        parsed = urlparse(url)
        return {
            # Domain detection - make configurable for different sites
            'is_target_domain': parsed.netloc == 'archive-it.org',  # Configure for your target domain
            'path_length': len(parsed.path),
            'has_query': bool(parsed.query),
            'query_length': len(parsed.query),
            'path_segments_count': len([s for s in parsed.path.split('/') if s]),
            'is_root_path': parsed.path in ['/', ''],
            'has_fragment': bool(parsed.fragment)
        }
    except:
        return {
            'is_target_domain': False,
            'path_length': 0,
            'has_query': False,
            'query_length': 0,
            'path_segments_count': 0,
            'is_root_path': True,
            'has_fragment': False
        }

def extract_text_features(text):
    """Extract meaningful features from link text"""
    if pd.isna(text) or text == '':
        return {
            'text_length': 0,
            'has_numbers': False,
            'has_special_chars': False,
            'word_count': 0,
            'is_empty_text': True,
            'contains_nav_words': False
        }
    
    text = str(text).lower()
    nav_words = ['next', 'previous', 'more', 'page', 'home', 'back', 'forward']
    
    return {
        'text_length': len(text),
        'has_numbers': bool(re.search(r'\d', text)),
        'has_special_chars': bool(re.search(r'[^a-zA-Z0-9\s]', text)),
        'word_count': len(text.split()),
        'is_empty_text': False,
        'contains_nav_words': any(word in text for word in nav_words)
    }

def group_source_pages(source_page):
    """Group source pages into meaningful categories"""
    if pd.isna(source_page):
        return 'unknown'
    
    if 'explore' in source_page and '?' not in source_page:
        return 'main_explore'
    elif 'explore' in source_page and '?' in source_page:
        return 'filtered_explore'
    elif 'organizations/' in source_page:
        return 'organization_detail'
    elif 'collections/' in source_page:
        return 'collection_detail'
    else:
        return 'other'

def remove_low_variance_features(df, threshold=0.05):
    """Remove boolean features with very low variance"""
    boolean_cols = df.select_dtypes(include=['bool']).columns
    low_variance_cols = []
    
    for col in boolean_cols:
        true_ratio = df[col].mean()
        if true_ratio < threshold or true_ratio > (1 - threshold):
            low_variance_cols.append(col)
            print(f"Removing low variance feature: {col} ({true_ratio:.1%} True)")
    
    return df.drop(columns=low_variance_cols), low_variance_cols

def preprocess_training_data(input_csv_path, output_csv_path):
    """
    Complete preprocessing pipeline for Random Forest training
    """
    print("Loading raw training data...")
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # 1. Handle missing values
    print("\n1. Handling missing values...")
    df['link_text'] = df['link_text'].fillna('NO_TEXT')
    df['show_param_value'] = df['show_param_value'].fillna('NONE')
    print(f"Filled missing values: link_text and show_param_value")
    
    # 2. Extract URL features (replace high-cardinality URL column)
    print("\n2. Extracting URL features...")
    url_features = df['url'].apply(extract_url_features)
    url_features_df = pd.DataFrame(url_features.tolist())
    
    # 3. Extract text features (replace high-cardinality text column)
    print("\n3. Extracting text features...")
    text_features = df['link_text'].apply(extract_text_features)
    text_features_df = pd.DataFrame(text_features.tolist())
    
    # 4. Group source pages (reduce cardinality)
    print("\n4. Grouping source pages...")
    df['source_page_type'] = df['source_page'].apply(group_source_pages)
    
    # 5. Encode categorical features
    print("\n5. Encoding categorical features...")
    # One-hot encode position (low cardinality)
    if 'position_on_page' in df.columns:
        position_dummies = pd.get_dummies(df['position_on_page'], prefix='position')
        df = pd.concat([df, position_dummies], axis=1)
    
    # Label encode show_param_value (low cardinality)
    if 'show_param_value' in df.columns:
        le_show = LabelEncoder()
        df['show_param_encoded'] = le_show.fit_transform(df['show_param_value'])
    
    # Label encode source_page_type
    le_source = LabelEncoder()
    df['source_page_type_encoded'] = le_source.fit_transform(df['source_page_type'])
    
    # 6. Combine all features
    print("\n6. Combining processed features...")
    # Drop original high-cardinality columns
    columns_to_drop = ['url', 'link_text', 'source_page', 'position_on_page', 'show_param_value', 'source_page_type']
    df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Add extracted features
    df_processed = pd.concat([df_processed, url_features_df, text_features_df], axis=1)
    
    # 7. Remove low variance features
    print("\n7. Removing low variance features...")
    df_processed, removed_features = remove_low_variance_features(df_processed)
    
    # 8. Final data quality check
    print("\n8. Final data quality check...")
    print(f"Final shape: {df_processed.shape}")
    print(f"Features: {len(df_processed.columns)}")
    print(f"Missing values: {df_processed.isnull().sum().sum()}")
    
    # Ensure we have our target variable
    if 'label_contextual' not in df_processed.columns:
        print("Warning: 'label_contextual' not found!")
    else:
        label_dist = df_processed['label_contextual'].mean()
        print(f"Label distribution: {label_dist:.1%} positive")
    
    # 9. Save processed data
    print(f"\n9. Saving processed data to {output_csv_path}")
    df_processed.to_csv(output_csv_path, index=False)
    
    # Save feature info for later use
    feature_info = {
        'removed_low_variance': removed_features,
        'label_encoders': {
            'show_param': le_show.classes_.tolist() if 'show_param_value' in df.columns else [],
            'source_page_type': le_source.classes_.tolist()
        },
        'final_features': df_processed.columns.tolist()
    }
    
    import json
    with open(str(output_csv_path).replace('.csv', '_feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("Preprocessing complete!")
    return df_processed

if __name__ == "__main__":
    input_path = Path("results/training_data_v2.csv")
    output_path = Path("results/training_data_processed.csv")
    
    processed_df = preprocess_training_data(input_path, output_path)