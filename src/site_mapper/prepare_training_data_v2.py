import json
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse, parse_qs

def extract_advanced_features(link_data):
    """
    Extract more nuanced features that capture context, not just simple boolean flags.
    This will help the ML model learn the edge cases that simple heuristics miss.
    """
    analysis = link_data.get('analysis', {})
    archive_analysis = analysis.get('analyze_archive_it_link', {})
    
    # Parse URL for detailed analysis
    parsed_url = urlparse(link_data['absolute_url'])
    query_params = parse_qs(parsed_url.query)
    path_segments = [seg for seg in parsed_url.path.split('/') if seg]
    
    features = {
        # Basic features
        'url': link_data['absolute_url'],
        'link_text': link_data['text'],
        'is_external': link_data['is_external'],
        
        # Context-aware pagination features
        'has_pagination': 'page' in query_params,
        'is_main_list_pagination': (
            'page' in query_params and 
            len(path_segments) <= 2 and  # shallow path = main lists
            'explore' in parsed_url.path
        ),
        'is_nested_pagination': (
            'page' in query_params and 
            len(path_segments) > 2  # deep path = might be less important
        ),
        
        # Context-aware show parameters
        'has_show_param': 'show' in query_params,
        'show_param_value': query_params.get('show', [''])[0],
        'is_view_toggle': query_params.get('show', [''])[0] in ['Collections', 'Organizations', 'Sites'],
        'is_detailed_view': query_params.get('show', [''])[0] in ['full_details', 'expanded'],
        
        # Path analysis for content type
        'path_depth': len(path_segments),
        'is_organization_detail': 'organizations' in path_segments and len(path_segments) >= 2,
        'is_collection_detail': 'collections' in path_segments and len(path_segments) >= 2,
        'is_main_explore': parsed_url.path.strip('/') == 'explore',
        
        # Query parameter complexity
        'num_query_params': len(query_params),
        'has_multiple_filters': sum(1 for key in query_params.keys() if key.startswith('f')) > 1,
        'has_sort_and_filter': 'sort' in query_params and any(k.startswith('f') for k in query_params.keys()),
        
        # Position and DOM context
        'position_on_page': analysis.get('link_position', ''),
        'in_navigation': any(parent in ['nav', 'header'] for parent in analysis.get('parent_elements', [])),
        'in_main_content': any(parent in ['main', 'content'] for parent in analysis.get('parent_elements', [])),
        
        # Link text analysis
        'text_length': len(link_data['text']),
        'is_navigation_text': any(word in link_data['text'].lower() for word in ['next', 'previous', 'page', 'more']),
        'is_action_text': any(word in link_data['text'].lower() for word in ['view', 'show', 'display', 'browse']),
        
        # Trap indicators (but more nuanced)
        'in_faceted_search': archive_analysis.get('in_faceted_search_ui', False),
        'has_sorting': 'sort' in query_params,
        'has_filtering': any(key.startswith('f') for key in query_params.keys()),
        'potential_issues_count': len(archive_analysis.get('potential_issues', [])),
        
        # Value indicators
        'leads_to_content': (
            ('organizations' in path_segments or 'collections' in path_segments) and
            len(path_segments) >= 2
        ),
        'is_essential_navigation': (
            link_data['text'].lower() in ['home', 'explore', 'browse', 'search'] or
            any(parent in ['nav', 'header'] for parent in analysis.get('parent_elements', []))
        )
    }
    
    return features

def create_contextual_labels(df):
    """
    Create more nuanced labels that consider context, not just simple rules.
    """
    # Essential pagination should be considered good
    essential_pagination = (
        df['is_main_list_pagination'] & 
        ~df['has_multiple_filters'] &  # Not overly complex
        (df['text_length'] < 20)  # Simple link text
    )
    
    # View toggles are sometimes necessary
    useful_view_toggles = (
        df['is_view_toggle'] & 
        df['in_main_content'] &  # In main content area
        ~df['in_faceted_search']  # Not in sidebar filters
    )
    
    # High-value content pages
    valuable_content = (
        df['leads_to_content'] |
        df['is_essential_navigation'] |
        (df['is_main_explore'] & ~df['has_multiple_filters'])
    )
    
    # Clear traps
    obvious_traps = (
        df['in_faceted_search'] |
        df['has_sort_and_filter'] |
        (df['has_multiple_filters'] & df['has_sorting']) |
        (df['potential_issues_count'] > 2)
    )
    
    # Create nuanced labels
    df['label_simple'] = valuable_content & ~obvious_traps  # Original simple approach
    df['label_contextual'] = (
        (valuable_content | essential_pagination | useful_view_toggles) & 
        ~obvious_traps
    )
    
    return df

def create_training_dataset_v2(json_path, output_csv_path):
    """
    Create an improved training dataset with contextual features and labels.
    """
    print(f"Loading data from {json_path}")
    with open(json_path, 'r') as f:
        crawl_data = json.load(f)
    
    # Extract features from all links
    all_links = []
    for source_url, links in crawl_data.items():
        for link in links:
            features = extract_advanced_features(link)
            features['source_page'] = source_url
            all_links.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_links)
    
    # Create contextual labels
    df = create_contextual_labels(df)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    # Print comparison statistics
    print("\nDataset Statistics:")
    print(f"Total links: {len(df)}")
    print("\nSimple vs Contextual Labeling:")
    print(f"Simple 'good' links: {df['label_simple'].sum()}")
    print(f"Contextual 'good' links: {df['label_contextual'].sum()}")
    print(f"Difference: {df['label_contextual'].sum() - df['label_simple'].sum()}")
    
    print("\nPagination Analysis:")
    print(f"Total pagination links: {df['has_pagination'].sum()}")
    print(f"Main list pagination: {df['is_main_list_pagination'].sum()}")
    print(f"Nested pagination: {df['is_nested_pagination'].sum()}")
    
    print("\nShow Parameter Analysis:")
    print(f"View toggles: {df['is_view_toggle'].sum()}")
    print(f"Links with show params: {df['has_show_param'].sum()}")
    
    print("\nSample of different labels:")
    sample_cols = ['url', 'label_simple', 'label_contextual', 'has_pagination', 'is_view_toggle']
    different_labels = df[df['label_simple'] != df['label_contextual']]
    if len(different_labels) > 0:
        print(different_labels[sample_cols].head())
    
    return df

if __name__ == "__main__":
    results_dir = Path("results")
    json_path = results_dir / "crawl_results_final.json"
    output_csv_path = results_dir / "training_data_v2.csv"
    
    df = create_training_dataset_v2(json_path, output_csv_path)