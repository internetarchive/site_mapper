"""
Inference-time ML link filter for the crawler.

Responsibilities:
- Load trained model and metadata (feature list)
- Extract the same features used in training from a link JSON
- Predict whether a link should be crawled (boolean + confidence)
- Track and save approved/rejected links for analysis

Inputs:
- results/model/random_forest_model.joblib
- results/model/model_info.json

Outputs (when save_filter_decisions is called):
- results/ml_rejected_links.json
- results/ml_approved_links.json

Typical usage:
  ml_filter = MLLinkFilter("results/model/random_forest_model.joblib", "results/model/model_info.json")
  should_crawl = ml_filter.should_crawl_link(link_data)
"""

import json
import logging
from urllib.parse import urlparse
import joblib
import pandas as pd
import re
from typing import Dict, Any


class MLLinkFilter:
    """
    Machine Learning based link filter for the crawler.

    Loads a trained classifier and accompanying metadata to make per-link
    crawl/skip decisions consistent with the training pipeline.
    """
    
    def __init__(self, model_path, model_info_path):
        """
        Load the trained model and its metadata.

        Args:
            model_path: Path to a Joblib file containing the fitted classifier
            model_info_path: Path to model_info.json with feature_names and metrics
        """
        self.model = joblib.load(model_path)
        
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.feature_names = self.model_info['feature_names']
        
        self.total_links_analyzed = 0
        self.links_filtered_out = 0
        self.links_approved = 0
        self.rejected_links = []
        self.approved_links = []
        
        logging.info(f"ML Filter loaded with {len(self.feature_names)} features")
    
    def extract_features_for_prediction(self, link_data):
        """
        Extract the feature vector for a single link using the same logic
        as the preprocessing pipeline.

        Args:
            link_data: dict representing one outlink with fields like
                absolute_url, text, is_external, and nested analysis

        Returns:
            list ordered to match self.feature_names, containing numeric/boolean
            features ready for prediction
        """

        url = link_data['absolute_url']
        try:
            parsed = urlparse(url)
            url_features = {
                'is_target_domain': parsed.netloc in ['archive-it.org'],
                'path_length': len(parsed.path),
                'has_query': bool(parsed.query),
                'query_length': len(parsed.query),
                'path_segments_count': len([s for s in parsed.path.split('/') if s]),
                'is_root_path': parsed.path in ['/', ''],
                'has_fragment': bool(parsed.fragment)
            }
        except:
            url_features = {
                'is_target_domain': False,
                'path_length': 0,
                'has_query': False,
                'query_length': 0,
                'path_segments_count': 0,
                'is_root_path': True,
                'has_fragment': False
            }
        

        text = link_data.get('text', '')
        if pd.isna(text) or text == '':
            text_features = {
                'text_length': 0,
                'has_numbers': False,
                'has_special_chars': False,
                'word_count': 0,
                'is_empty_text': True,
                'contains_nav_words': False
            }
        else:
            text = str(text).lower()
            nav_words = ['next', 'previous', 'more', 'page', 'home', 'back', 'forward']
            text_features = {
                'text_length': len(text),
                'has_numbers': bool(re.search(r'\d', text)),
                'has_special_chars': bool(re.search(r'[^a-zA-Z0-9\s]', text)),
                'word_count': len(text.split()),
                'is_empty_text': False,
                'contains_nav_words': any(word in text for word in nav_words)
            }
        

        analysis = link_data.get('analysis', {})
        archive_analysis = analysis.get('analyze_archive_it_link', {})
        
        features = {
            'is_external': link_data.get('is_external', False),
            'has_show_param': archive_analysis.get('has_show_param', False),
            'is_view_toggle': archive_analysis.get('has_show_param', False),
            'is_organization_detail': '/organizations/' in url and len(url.split('/')) >= 4,
            'is_collection_detail': '/collections/' in url and len(url.split('/')) >= 4,
            'is_main_explore': urlparse(url).path.strip('/') == 'explore',
            'num_query_params': len(archive_analysis.get('query_params', [])),
            'has_sort_and_filter': False,
            'in_main_content': False,
            'is_action_text': any(word in text.lower() for word in ['view', 'show', 'display', 'browse']),
            'has_sorting': 'sort' in archive_analysis.get('query_params', []),
            'has_filtering': any(param.startswith('f') for param in archive_analysis.get('query_params', [])),
            'leads_to_content': (
                ('/organizations/' in url or '/collections/' in url) and
                len(url.split('/')) >= 4
            ),
            'is_essential_navigation': text.lower() in ['home', 'explore', 'browse', 'search'],
            'path_depth': len([s for s in urlparse(url).path.split('/') if s]),
        }
        
        # Combine features
        all_features = {**url_features, **text_features, **features}
        
        # Create feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(all_features.get(feature_name, 0))
        
        return feature_vector
    
    def should_crawl_link(self, link_data: Dict[str, Any]) -> bool:
        """
        Use the ML model to decide whether a link should be crawled.

        Args:
            link_data: outlink record emitted by the crawler

        Returns:
            True if the link is predicted to be valuable (crawl), False otherwise
        """
        self.total_links_analyzed += 1
        
        try:
            features = self.extract_features_for_prediction(link_data)
            
            # Convert to pandas DataFrame with proper feature names to avoid warning
            feature_df = pd.DataFrame([features], columns=self.feature_names)
            
            prediction = self.model.predict(feature_df)[0]
            confidence = self.model.predict_proba(feature_df)[0]
            
            should_crawl = bool(prediction)
            
            if should_crawl:
                self.links_approved += 1
                # Store approved link details
                self.approved_links.append({
                    'url': link_data['absolute_url'],
                    'text': link_data.get('text', ''),
                    'confidence': confidence[1]  # Confidence for "good" class
                })
            else:
                self.links_filtered_out += 1

                # Store rejected link details
                self.rejected_links.append({
                    'url': link_data['absolute_url'],
                    'text': link_data.get('text', ''),
                    'confidence': confidence[0],  # Confidence for "bad" class
                    'reason': 'ML_MODEL_PREDICTION'
                })

                logging.info(f"ML FILTER: Rejected {link_data['absolute_url']} "
                           f"(confidence: {confidence[0]:.3f})")
            
            return should_crawl
            
        except Exception as e:
            logging.error(f"ML Filter error for {link_data['absolute_url']}: {e}")
            self.links_approved += 1
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregate filtering statistics for the current crawl session.

        Returns:
            dict with counts for total_analyzed, approved, filtered_out, filter_rate
        """
        return {
            'total_analyzed': self.total_links_analyzed,
            'approved': self.links_approved,
            'filtered_out': self.links_filtered_out,
            'filter_rate': self.links_filtered_out / max(self.total_links_analyzed, 1)
        }
    
    def save_filter_decisions(self, output_dir: str):
        """
        Save all approved/rejected link decisions for offline analysis.

        Args:
            output_dir: directory where JSON files will be written

        Writes:
            ml_rejected_links.json, ml_approved_links.json in output_dir
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        rejected_path = os.path.join(output_dir, 'ml_rejected_links.json')
        with open(rejected_path, 'w') as f:
            json.dump(self.rejected_links, f, indent=2)
        
        approved_path = os.path.join(output_dir, 'ml_approved_links.json')
        with open(approved_path, 'w') as f:
            json.dump(self.approved_links, f, indent=2)
        
        print(f"Filter decisions saved:")
        print(f"  Rejected links: {rejected_path}")
        print(f"  Approved links: {approved_path}")
        print(f"  Total rejected: {len(self.rejected_links):,}")
        print(f"  Total approved: {len(self.approved_links):,}")