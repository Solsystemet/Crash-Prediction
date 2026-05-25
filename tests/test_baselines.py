"""Quick test for baseline module."""

import numpy as np
<<<<<<< HEAD
from training.baselines import (
    ClassificationBaseline,
    RegressionBaseline,
    compare_to_baseline,
    create_imbalance_baselines,
    STRATEGY_ALIASES,
)
=======
from training.baselines import ClassificationBaseline, RegressionBaseline, compare_to_baseline
>>>>>>> origin/dev

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


<<<<<<< HEAD
def test_coin_flip_baselines():
    """Test coin flip baseline aliases for imbalanced classification."""
    # Simulate imbalanced data: 70% class 0, 20% class 1, 10% class 2
    np.random.seed(42)
    y_train = np.array([0] * 700 + [1] * 200 + [2] * 100)
    y_test = np.array([0] * 70 + [1] * 20 + [2] * 10)
    
    # Test using aliases directly
    baseline = ClassificationBaseline(
        strategies=['coin_flip', 'biased_coin_flip'],
        random_state=42
    )
    baseline.fit(y_train)
    results = baseline.evaluate(y_test, class_names=['NO_INJURY', 'MINOR', 'SEVERE'])
    
    print('\nCoin flip baseline results:')
    for strategy, result in results.items():
        print(f'  {strategy}: accuracy={result.metrics["accuracy"]:.3f}, f1_macro={result.metrics["f1_macro"]:.3f}')
    
    # Verify aliases are in results
    assert 'coin_flip' in results, "coin_flip alias should be in results"
    assert 'biased_coin_flip' in results, "biased_coin_flip alias should be in results"
    
    # Verify aliases map correctly
    assert STRATEGY_ALIASES['coin_flip'] == 'uniform'
    assert STRATEGY_ALIASES['biased_coin_flip'] == 'stratified'
    
    print('  Coin flip baselines: OK')


def test_create_imbalance_baselines():
    """Test the convenience function for imbalance baselines."""
    # Simulate imbalanced data
    y_train = np.array([0] * 500 + [1] * 100 + [2] * 50)
    y_test = np.array([0] * 50 + [1] * 10 + [2] * 5)
    
    baseline = create_imbalance_baselines(random_state=42)
    baseline.fit(y_train)
    results = baseline.evaluate(y_test, class_names=['NO_INJURY', 'MINOR', 'SEVERE'])
    
    print('\nImbalance baselines (via create_imbalance_baselines):')
    for strategy, result in results.items():
        print(f'  {strategy}: accuracy={result.metrics["accuracy"]:.3f}')
    
    assert 'coin_flip' in results
    assert 'biased_coin_flip' in results
    
    # biased_coin_flip should predict class 0 more often (matches training distribution)
    # This is a probabilistic test, so we use many samples
    predictions = baseline.predict(10000, strategy='biased_coin_flip')
    class_0_ratio = np.mean(predictions == 0)
    
    # Should be roughly 500/650 ≈ 0.77 (with some variance)
    assert 0.65 < class_0_ratio < 0.85, f"biased_coin_flip should reflect class distribution, got {class_0_ratio:.3f}"
    
    # coin_flip should be roughly uniform (1/3 each)
    predictions_uniform = baseline.predict(10000, strategy='coin_flip')
    class_0_ratio_uniform = np.mean(predictions_uniform == 0)
    assert 0.28 < class_0_ratio_uniform < 0.40, f"coin_flip should be uniform, got {class_0_ratio_uniform:.3f}"
    
    print('  create_imbalance_baselines: OK')


=======
>>>>>>> origin/dev
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
<<<<<<< HEAD
    test_coin_flip_baselines()
    test_create_imbalance_baselines()
=======
>>>>>>> origin/dev
    test_regression_baseline()
    test_compare_to_baseline()
    print('\n✓ All baseline tests passed!')
