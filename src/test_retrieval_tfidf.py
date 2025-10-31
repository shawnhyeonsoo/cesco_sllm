"""
Test Retrieval Performance using TF-IDF for Category Classification

This script evaluates category retrieval using traditional TF-IDF approach.
Much more memory-efficient than embedding models - no large models to load!
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetrievalTester:
    """Test retrieval performance using TF-IDF for category classification."""
    
    def __init__(
        self,
        test_data_path: str,
        categories_path: str,
        max_samples: int = None,
        top_k: int = 5,
        max_features: int = 5000
    ):
        """
        Initialize the TF-IDF retrieval tester.
        
        Args:
            test_data_path: Path to test dataset JSON file
            categories_path: Path to categories CSV file
            max_samples: Maximum number of samples to test (None = all)
            top_k: Number of top categories to retrieve
            max_features: Maximum number of TF-IDF features (reduce if memory issues)
        """
        print(f"Initializing TF-IDF Retrieval Tester")
        print(f"Max features: {max_features}")
        
        self.max_samples = max_samples
        self.top_k = top_k
        self.max_features = max_features
        
        # Load test data
        print(f"\nLoading test data from {test_data_path}...")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        if max_samples:
            self.test_data = self.test_data[:max_samples]
        print(f"Loaded {len(self.test_data)} test samples")
        
        # Load categories
        print(f"\nLoading categories from {categories_path}...")
        self.categories_df = pd.read_csv(categories_path)
        self.categories = self._parse_categories()
        print(f"Loaded {len(self.categories)} categories")
        
        # Initialize TF-IDF vectorizer
        print("\nInitializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Fit vectorizer and compute category vectors
        print("Computing TF-IDF vectors for categories...")
        self.category_texts = self._create_category_texts()
        self.category_vectors = self.vectorizer.fit_transform(self.category_texts)
        print(f"TF-IDF matrix shape: {self.category_vectors.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print("✓ Initialization complete!")
        
    def _parse_categories(self) -> List[str]:
        """Parse categories from DataFrame into list format."""
        categories = []
        for _, row in self.categories_df.iterrows():
            if pd.notna(row.get('대분류')) and pd.notna(row.get('중분류')) and pd.notna(row.get('소분류')):
                category = f"{row['대분류']}__{row['중분류']}__{row['소분류']}"
                categories.append(category)
        return sorted(list(set(categories)))
    
    def _create_category_texts(self) -> List[str]:
        """Create text representations of categories for TF-IDF."""
        category_texts = []
        for category in self.categories:
            parts = category.split('__')
            # Combine all parts with spaces for better TF-IDF
            # Repeat parts to give them more weight
            text = f"{parts[0]} {parts[0]} {parts[1]} {parts[1]} {parts[2]} {parts[2]}"
            category_texts.append(text)
        return category_texts
    
    def retrieve_categories(self, input_text: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar categories for input text.
        
        Args:
            input_text: Customer VOC text
            
        Returns:
            List of (category, similarity_score) tuples
        """
        # Transform input text to TF-IDF vector
        input_vector = self.vectorizer.transform([input_text])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(input_vector, self.category_vectors)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Return categories with scores
        return [(self.categories[idx], similarities[idx]) for idx in top_k_indices]
    
    def calculate_metrics(
        self,
        predicted: List[str],
        ground_truth: List[str],
        similarity_scores: List[float] = None
    ) -> Dict[str, float]:
        """
        Calculate retrieval metrics.
        
        Args:
            predicted: List of predicted categories
            ground_truth: List of ground truth categories
            similarity_scores: Optional list of similarity scores
            
        Returns:
            Dictionary of metrics
        """
        pred_set = set(predicted)
        gt_set = set(ground_truth)
        
        true_positives = len(pred_set.intersection(gt_set))
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        # Precision
        precision = true_positives / len(pred_set) if pred_set else 0.0
        
        # Recall
        recall = true_positives / len(gt_set) if gt_set else 0.0
        
        # F1 Score
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # Exact Match
        exact_match = 1.0 if pred_set == gt_set else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, pred_cat in enumerate(predicted):
            if pred_cat in gt_set:
                mrr = 1.0 / (i + 1)
                break
        
        # Average similarity score
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': exact_match,
            'mrr': mrr,
            'avg_similarity': avg_similarity,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def calculate_recall_at_k(self, all_results: List[Dict]) -> Dict[str, float]:
        """
        Calculate Recall@K for different K values.
        
        Args:
            all_results: List of result dictionaries
            
        Returns:
            Dictionary of Recall@K metrics
        """
        recall_at_k = {}
        
        for k in [1, 3, 5]:
            if k > self.top_k:
                continue
                
            recalls = []
            for result in all_results:
                predicted = result['predicted'][:k]
                ground_truth = set(result['ground_truth'])
                
                hits = len(set(predicted).intersection(ground_truth))
                recall = hits / len(ground_truth) if ground_truth else 0.0
                recalls.append(recall)
            
            recall_at_k[f'recall@{k}'] = np.mean(recalls)
        
        return recall_at_k
    
    def evaluate(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Evaluate retrieval performance on test data.
        
        Returns:
            Tuple of (aggregate metrics, detailed results DataFrame)
        """
        print(f"\nEvaluating on {len(self.test_data)} samples...")
        print(f"Retrieving top-{self.top_k} categories for each sample...")
        
        all_metrics = []
        detailed_results = []
        
        for idx, sample in enumerate(tqdm(self.test_data, desc="Evaluating")):
            input_text = sample.get('input', '')
            ground_truth = sample.get('categories', [])
            
            # Retrieve categories
            retrieved = self.retrieve_categories(input_text)
            predicted = [cat for cat, score in retrieved]
            similarity_scores = [score for cat, score in retrieved]
            
            # Calculate metrics
            metrics = self.calculate_metrics(predicted, ground_truth, similarity_scores)
            all_metrics.append(metrics)
            
            # Store detailed results
            detailed_results.append({
                'sample_id': idx,
                'input': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'similarity_scores': similarity_scores,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'exact_match': metrics['exact_match'],
                'mrr': metrics['mrr'],
                'avg_similarity': metrics['avg_similarity'],
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives']
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1': np.mean([m['f1'] for m in all_metrics]),
            'exact_match_accuracy': np.mean([m['exact_match'] for m in all_metrics]),
            'mrr': np.mean([m['mrr'] for m in all_metrics]),
            'avg_similarity': np.mean([m['avg_similarity'] for m in all_metrics]),
            'avg_true_positives': np.mean([m['true_positives'] for m in all_metrics]),
            'avg_false_positives': np.mean([m['false_positives'] for m in all_metrics]),
            'avg_false_negatives': np.mean([m['false_negatives'] for m in all_metrics])
        }
        
        # Calculate Recall@K
        recall_at_k = self.calculate_recall_at_k(detailed_results)
        aggregate_metrics.update(recall_at_k)
        
        results_df = pd.DataFrame(detailed_results)
        
        return aggregate_metrics, results_df
    
    def print_results(self, metrics: Dict[str, float]):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("TF-IDF RETRIEVAL PERFORMANCE RESULTS")
        print("="*60)
        print(f"\nMethod: TF-IDF")
        print(f"Test Samples: {len(self.test_data)}")
        print(f"Top-K: {self.top_k}")
        print(f"Max Features: {self.max_features}")
        print(f"\nAggregate Metrics:")
        print(f"  Precision:           {metrics['precision']:.4f}")
        print(f"  Recall:              {metrics['recall']:.4f}")
        print(f"  F1 Score:            {metrics['f1']:.4f}")
        print(f"  Exact Match Acc:     {metrics['exact_match_accuracy']:.4f}")
        print(f"  MRR:                 {metrics['mrr']:.4f}")
        print(f"  Avg Similarity:      {metrics['avg_similarity']:.4f}")
        
        # Print Recall@K
        print(f"\nRecall@K Metrics:")
        for k in [1, 3, 5]:
            key = f'recall@{k}'
            if key in metrics:
                print(f"  Recall@{k}:            {metrics[key]:.4f}")
        
        print(f"\nAverage Counts per Sample:")
        print(f"  True Positives:      {metrics['avg_true_positives']:.2f}")
        print(f"  False Positives:     {metrics['avg_false_positives']:.2f}")
        print(f"  False Negatives:     {metrics['avg_false_negatives']:.2f}")
        print("="*60)
    
    def save_results(
        self,
        metrics: Dict[str, float],
        results_df: pd.DataFrame,
        output_dir: str = "data"
    ):
        """
        Save evaluation results to files.
        
        Args:
            metrics: Aggregate metrics dictionary
            results_df: Detailed results DataFrame
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate metrics
        metrics_file = output_path / f"tfidf_retrieval_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nSaved aggregate metrics to: {metrics_file}")
        
        # Save detailed results
        results_file = output_path / f"tfidf_retrieval_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"Saved detailed results to: {results_file}")
        
        # Save summary report
        report_file = output_path / f"tfidf_retrieval_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("TF-IDF RETRIEVAL PERFORMANCE REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Method: TF-IDF\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Samples: {len(self.test_data)}\n")
            f.write(f"Total Categories: {len(self.categories)}\n")
            f.write(f"Top-K: {self.top_k}\n")
            f.write(f"Max Features: {self.max_features}\n\n")
            f.write("Aggregate Metrics:\n")
            f.write("-"*60 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name:.<50} {value:.4f}\n")
            f.write("\n" + "="*60 + "\n")
        print(f"Saved summary report to: {report_file}")


def main():
    """Main function to run TF-IDF retrieval performance testing."""
    parser = argparse.ArgumentParser(
        description="Test retrieval performance using TF-IDF for category classification"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/processed_test_dataset.json",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="assets/voc_category_final.csv",
        help="Path to categories CSV"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to test (default: all)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top categories to retrieve (default: 5)"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features (default: 5000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TF-IDF RETRIEVAL PERFORMANCE TESTING")
    print("="*60)
    print("\nℹ️  TF-IDF is a lightweight, memory-efficient baseline")
    print("   No large models required!")
    print("="*60)
    
    # Initialize tester
    tester = TfidfRetrievalTester(
        test_data_path=args.test_data,
        categories_path=args.categories,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_features=args.max_features
    )
    
    # Run evaluation
    metrics, results_df = tester.evaluate()
    
    # Print results
    tester.print_results(metrics)
    
    # Save results
    tester.save_results(metrics, results_df, args.output_dir)
    
    print("\n✅ Evaluation completed successfully!")
    print("\n💡 TIP: Compare these baseline results with embedding models")
    print("   to see how much improvement embeddings provide!")


if __name__ == "__main__":
    main()
