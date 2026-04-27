"""Quick test for baseline module."""

import numpy as np
from training.baselines import ClassificationBaseline, RegressionBaseline, compare_to_baseline

def test_classification_baseline():
    """Test classification baseline."""
    y_train = np.array([0, 0, 0, 1, 1, 2])  # 3 classes
    y_test = np.array([0, 1, 2, 0, 1])

    baseline = ClassificationBaseline(strategies=['most_frequent', 'stratified'], random_state=42)
    baseline.fit(y_train)
    results = baseline.evaluate(y_test, class_names=['A', 'B', 'C'])
    
    print('Classification baseline results:')
    for strategy, result in results.items():
        print(f'  {strategy}: accuracy={result.metrics["accuracy"]:.3f}')
    
    assert 'most_frequent' in results
    assert 'stratified' in results
    print('  Classification baseline: OK')


def test_regression_baseline():
    """Test regression baseline."""
    y_train_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_test_reg = np.array([2.0, 3.0, 4.0])

    baseline_reg = RegressionBaseline(strategies=['mean', 'median'])
    baseline_reg.fit(y_train_reg)
    results_reg = baseline_reg.evaluate(y_test_reg)
    
    print('\nRegression baseline results:')
    for strategy, result in results_reg.items():
        print(f'  {strategy}: mae={result.metrics["mae"]:.3f}, r2={result.metrics["r2"]:.3f}')
    
    assert 'mean' in results_reg
    assert 'median' in results_reg
    print('  Regression baseline: OK')


def test_compare_to_baseline():
    """Test comparison function."""
    model_metrics = {'mae': 1.0, 'r2': 0.8}
    
    # Create fake baseline results
    from training.baselines import BaselineResult
    baseline_results = {
        'mean': BaselineResult(strategy='mean', metrics={'mae': 2.0, 'r2': 0.0})
    }
    
    comparison = compare_to_baseline(model_metrics, baseline_results)
    
    assert 'mean' in comparison
    assert comparison['mean']['mae'] > 0  # Model is better (lower MAE)
    print('\n  Comparison: OK')


if __name__ == '__main__':
    test_classification_baseline()
    test_regression_baseline()
    test_compare_to_baseline()
    print('\n✓ All baseline tests passed!')
