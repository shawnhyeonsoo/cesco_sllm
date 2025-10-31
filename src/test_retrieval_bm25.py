"""
Test Retrieval Performance using BM25 for Category Classification

BM25 is an improved ranking function over TF-IDF that considers document length
and uses better term frequency saturation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
from datetime import datetime
from rank_bm25 import BM25Okapi
import re


class BM25RetrievalTester:
    """Test retrieval performance using BM25 for category classification."""
    
    def __init__(
        self,
        test_data_path: str,
        categories_path: str,
        max_samples: int = None,
        top_k: int = 5
    ):
        """
        Initialize the BM25 retrieval tester.
        
        Args:
            test_data_path: Path to test dataset JSON file
            categories_path: Path to categories CSV file
            max_samples: Maximum number of samples to test (None = all)
            top_k: Number of top categories to retrieve
        """
        print(f"Initializing BM25 Retrieval Tester")
        
        self.max_samples = max_samples
        self.top_k = top_k
        
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
        
        # Create BM25 index
        print("\nCreating BM25 index...")
        self.category_texts = self._create_category_texts()
        tokenized_corpus = [self._tokenize(text) for text in self.category_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✓ BM25 index created!")
        
    def _parse_categories(self) -> List[str]:
        """Parse categories from DataFrame into list format."""
        categories = []
        for _, row in self.categories_df.iterrows():
            if pd.notna(row.get('대분류')) and pd.notna(row.get('중분류')) and pd.notna(row.get('소분류')):
                category = f"{row['대분류']}__{row['중분류']}__{row['소분류']}"
                categories.append(category)
        return sorted(list(set(categories)))
    
    def _create_category_texts(self) -> List[str]:
        """Create text representations of categories."""
        category_texts = []
        for category in self.categories:
            parts = category.split('__')
            # Repeat parts for emphasis
            text = f"{parts[0]} {parts[0]} {parts[1]} {parts[1]} {parts[2]} {parts[2]}"
            category_texts.append(text)
        return category_texts
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for Korean/mixed text."""
        # Split on whitespace and keep Korean characters
        tokens = text.split()
        return [token for token in tokens if token.strip()]
    
    def retrieve_categories(self, input_text: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar categories using BM25.
        
        Args:
            input_text: Customer VOC text
            
        Returns:
            List of (category, score) tuples
        """
        tokenized_query = self._tokenize(input_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[-self.top_k:][::-1]
        
        # Return categories with scores
        return [(self.categories[idx], scores[idx]) for idx in top_k_indices]
    
    def calculate_metrics(
        self,
        predicted: List[str],
        ground_truth: List[str],
        scores: List[float] = None
    ) -> Dict[str, float]:
        """Calculate retrieval metrics."""
        pred_set = set(predicted)
        gt_set = set(ground_truth)
        
        true_positives = len(pred_set.intersection(gt_set))
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(gt_set) if gt_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        exact_match = 1.0 if pred_set == gt_set else 0.0
        
        mrr = 0.0
        for i, pred_cat in enumerate(predicted):
            if pred_cat in gt_set:
                mrr = 1.0 / (i + 1)
                break
        
        avg_score = np.mean(scores) if scores else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': exact_match,
            'mrr': mrr,
            'avg_score': avg_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def calculate_recall_at_k(self, all_results: List[Dict]) -> Dict[str, float]:
        """Calculate Recall@K metrics."""
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
        """Evaluate retrieval performance."""
        print(f"\nEvaluating on {len(self.test_data)} samples...")
        
        all_metrics = []
        detailed_results = []
        
        for idx, sample in enumerate(tqdm(self.test_data, desc="Evaluating")):
            input_text = sample.get('input', '')
            ground_truth = sample.get('categories', [])
            
            retrieved = self.retrieve_categories(input_text)
            predicted = [cat for cat, score in retrieved]
            scores = [score for cat, score in retrieved]
            
            metrics = self.calculate_metrics(predicted, ground_truth, scores)
            all_metrics.append(metrics)
            
            detailed_results.append({
                'sample_id': idx,
                'input': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'scores': scores,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'exact_match': metrics['exact_match'],
                'mrr': metrics['mrr'],
                'avg_score': metrics['avg_score'],
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives']
            })
        
        aggregate_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1': np.mean([m['f1'] for m in all_metrics]),
            'exact_match_accuracy': np.mean([m['exact_match'] for m in all_metrics]),
            'mrr': np.mean([m['mrr'] for m in all_metrics]),
            'avg_score': np.mean([m['avg_score'] for m in all_metrics]),
            'avg_true_positives': np.mean([m['true_positives'] for m in all_metrics]),
            'avg_false_positives': np.mean([m['false_positives'] for m in all_metrics]),
            'avg_false_negatives': np.mean([m['false_negatives'] for m in all_metrics])
        }
        
        recall_at_k = self.calculate_recall_at_k(detailed_results)
        aggregate_metrics.update(recall_at_k)
        
        results_df = pd.DataFrame(detailed_results)
        
        return aggregate_metrics, results_df
    
    def print_results(self, metrics: Dict[str, float]):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("BM25 RETRIEVAL PERFORMANCE RESULTS")
        print("="*60)
        print(f"\nMethod: BM25")
        print(f"Test Samples: {len(self.test_data)}")
        print(f"Top-K: {self.top_k}")
        print(f"\nAggregate Metrics:")
        print(f"  Precision:           {metrics['precision']:.4f}")
        print(f"  Recall:              {metrics['recall']:.4f}")
        print(f"  F1 Score:            {metrics['f1']:.4f}")
        print(f"  Exact Match Acc:     {metrics['exact_match_accuracy']:.4f}")
        print(f"  MRR:                 {metrics['mrr']:.4f}")
        print(f"  Avg Score:           {metrics['avg_score']:.4f}")
        
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
    
    def save_results(self, metrics: Dict[str, float], results_df: pd.DataFrame, output_dir: str = "data"):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metrics_file = output_path / f"bm25_retrieval_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics to: {metrics_file}")
        
        results_file = output_path / f"bm25_retrieval_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"Saved results to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Test BM25 retrieval performance")
    parser.add_argument("--test_data", type=str, default="data/processed_test_dataset.json")
    parser.add_argument("--categories", type=str, default="assets/voc_category_final.csv")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="data")
    
    args = parser.parse_args()
    
    print("="*60)
    print("BM25 RETRIEVAL PERFORMANCE TESTING")
    print("="*60)
    print("\n✓ BM25 usually outperforms TF-IDF!")
    print("="*60)
    
    tester = BM25RetrievalTester(
        test_data_path=args.test_data,
        categories_path=args.categories,
        max_samples=args.max_samples,
        top_k=args.top_k
    )
    
    metrics, results_df = tester.evaluate()
    tester.print_results(metrics)
    tester.save_results(metrics, results_df, args.output_dir)
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
