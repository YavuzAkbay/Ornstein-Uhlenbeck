# Ornstein-Uhlenbeck Process Analyzer: Advanced Mean Reversion Analysis for Commodities

**Author:** Yavuz Akbay  
**Email:** akbay.yavuz@gmail.com

A comprehensive quantitative analysis framework that implements the Ornstein-Uhlenbeck (OU) process to model mean-reverting behavior in commodity markets. This tool combines rigorous mathematical foundations with practical trading applications, providing scientific-grade analysis for quantitative analysts, traders, and researchers.

## 👨‍💻 Author

**Yavuz** - Quantitative Analyst

* 🔗 **LinkedIn**: <https://www.linkedin.com/in/yavuzakbay/>
* 📧 **Email**: akbay.yavuz@gmail.com
* 🐙 **GitHub**: <https://github.com/YavuzAkbay>

## 🎯 Executive Summary

In this project, I developed a sophisticated Ornstein-Uhlenbeck process analyzer specifically designed for commodity markets. The OU process is particularly valuable for commodities due to their inherent mean-reverting characteristics driven by supply-demand fundamentals, seasonal patterns, and storage costs. This implementation provides both theoretical rigor and practical trading applications, making it an essential tool for quantitative analysts working in commodity markets.

## 📚 Theoretical Foundation

### 🔄 Mean Reversion in Commodity Markets

Mean reversion in commodity markets is driven by several fundamental factors:

1. **Supply-Demand Dynamics**: Commodity prices naturally revert to levels that balance supply and demand
2. **Storage Costs**: Physical storage costs create upper bounds on price deviations
3. **Seasonal Patterns**: Agricultural and energy commodities exhibit seasonal mean reversion
4. **Arbitrage Opportunities**: Price deviations create profitable arbitrage opportunities that drive reversion

### 🧮 Ornstein-Uhlenbeck Process Mathematics

The OU process is a continuous-time stochastic differential equation:

```
dX(t) = θ(μ - X(t))dt + σdW(t)
```

Where:
- **θ (theta)**: Speed of mean reversion (annual rate)
- **μ (mu)**: Long-term mean level
- **σ (sigma)**: Volatility parameter
- **W(t)**: Wiener process (Brownian motion)

### 🔬 Key Mathematical Properties

1. **Mean Reversion Speed**: θ determines how quickly prices return to the long-term mean
2. **Half-Life**: Time for half the deviation to be eliminated: `t₁/₂ = ln(2)/θ`
3. **Stationary Distribution**: Normal distribution with mean μ and variance σ²/(2θ)
4. **Autocorrelation**: Exponential decay with rate θ

## ⚙️ Technical Specifications

### 📦 Dependencies

#### Core Dependencies
* **NumPy 1.21.0+**: Numerical computations and linear algebra
* **Pandas 1.3.0+**: Data manipulation and time series analysis
* **Matplotlib 3.5.0+**: Professional visualization and charting
* **SciPy 1.7.0+**: Scientific computing and optimization
* **Seaborn 0.11.0+**: Statistical visualization and styling
* **yfinance 0.1.70+**: Real-time commodity data fetching

#### Machine Learning Dependencies
* **PyTorch 1.12.0+**: Deep learning framework for parameter optimization
* **scikit-learn 1.1.0+**: Machine learning utilities and preprocessing

### 🚀 Performance Features

#### Traditional OU Analyzer
* **Customizable Parameters**: Flexible OU parameter specification
* **Multi-Commodity Analysis**: Batch processing of multiple commodities
* **Professional Visualizations**: Publication-ready charts and analysis
* **Statistical Validation**: Comprehensive model diagnostics
* **Real-time Data**: Live commodity price feeds via yfinance

#### Machine Learning Enhanced Features
* **🧠 Neural Network Optimization**: PyTorch-based parameter optimization
* **🎯 Goal Validation**: Multi-objective optimization with statistical testing
* **📊 Data Caching**: Intelligent caching system for faster analysis
* **🔄 Model Persistence**: Save and reuse trained models
* **⚡ GPU Acceleration**: CUDA support for faster training
* **📈 Advanced Visualizations**: ML-enhanced charts with optimization results

## 🔬 Research Applications

### 🎓 Academic Research

1. **Mean Reversion Studies**: Quantifying mean reversion in commodity markets
2. **Market Efficiency**: Testing weak-form market efficiency in commodity futures
3. **Risk Management**: Dynamic hedging strategies based on OU parameters
4. **Portfolio Optimization**: Mean reversion-based asset allocation

### 🏢 Industry Applications

1. **Commodity Trading**: Mean reversion trading strategies
2. **Risk Management**: Dynamic position sizing based on mean reversion strength
3. **Portfolio Management**: Commodity allocation in diversified portfolios
4. **Hedging Strategies**: Optimal hedging ratios using OU parameters

## 🧠 Machine Learning Enhanced OU Analysis

### 🎯 Goal Validation System

The ML-enhanced analyzer includes a sophisticated goal validation system that ensures your optimization objectives are truly optimal:

#### **8 Alternative Objective Functions**
1. **Pure Mean Reversion Strength** - Focus solely on mean reversion
2. **Sharpe Ratio** - Risk-adjusted returns optimization
3. **Volatility-Adjusted Mean Reversion** - Mean reversion per unit of volatility
4. **Band Hit Rate Optimization** - Optimal frequency of band crossings
5. **Consistency Score** - Low variance in returns
6. **Information Ratio** - Excess return per tracking error
7. **Calmar Ratio** - Return per maximum drawdown
8. **Maximum Drawdown Minimization** - Risk-focused optimization

#### **Statistical Validation**
- **T-tests** for statistical significance
- **Cohen's d** for effect size measurement
- **P-value analysis** for confidence levels
- **Cross-validation** across parameter combinations

#### **Backtesting Integration**
- **Strategy performance comparison** between objectives
- **Win rate analysis** for different approaches
- **Real trading simulation** results
- **Risk-adjusted performance** metrics

### 🔬 ML Architecture

#### **Neural Network Structure**
- **Input Layer**: Price statistics + parameter combinations
- **Hidden Layers**: Configurable (default: 128, 64, 32 neurons)
- **Output Layer**: Optimization score prediction
- **Regularization**: Batch normalization + dropout
- **Activation**: ReLU activation functions

#### **Training Process**
1. **Data Generation**: Creates 1000+ training samples with random parameter combinations
2. **Feature Engineering**: Combines price statistics with parameters
3. **Model Training**: Trains neural network to predict optimization scores
4. **Parameter Search**: Uses trained model to find optimal parameters
5. **Validation**: Calculates detailed metrics for optimal parameters

### 📊 ML Performance Metrics

#### **Optimization Results**
- **ML Optimization Score**: Neural network predicted score
- **Parameter Convergence**: Stability of optimal parameters
- **Cross-validation Accuracy**: Model generalization performance
- **Statistical Significance**: Confidence in optimization results

#### **Goal Validation Results**
- **Objective Comparison**: Performance across 8 different objectives
- **Improvement Percentage**: Quantified benefit of alternative objectives
- **Statistical Significance**: P-values and effect sizes
- **Backtest Performance**: Real trading strategy results

### 🎨 ML Visualizations

#### **Enhanced Charts**
- **ML-Optimized Parameters**: Shows neural network selected parameters
- **Optimization Score**: Displays ML confidence in parameter selection
- **Goal Validation**: Comprehensive objective comparison charts
- **Statistical Tests**: P-values and significance indicators

#### **Output Files**
- **`ml_ou_analysis_[SYMBOL].png`**: ML-enhanced analysis charts
- **`goal_validation_[SYMBOL].png`**: Goal validation results
- **`models/`**: Saved PyTorch models for reuse
- **`data_cache/`**: Cached historical data

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/YavuzAkbay/Ornstein-Uhlenbeck.git
cd Ornstein-Uhlenbeck

# Install dependencies
pip install -r requirements.txt
```

### Dependencies Installation

#### Core Dependencies
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.5.0 scipy>=1.7.0 seaborn>=0.11.0 yfinance>=0.1.70
```

#### Machine Learning Dependencies
```bash
pip install torch>=1.12.0 scikit-learn>=1.1.0
```

#### Complete Installation
```bash
pip install -r requirements.txt
```

## 🚀 Usage Examples

### Basic Analysis (Traditional OU)
```python
from calibrated_ou_analyzer import GenericOUAnalyzer

# Initialize analyzer with default parameters
analyzer = GenericOUAnalyzer()

# Analyze a single commodity
results = analyzer.analyze_commodity('GC=F', period='5y')

# Create visualization
fig = analyzer.create_visualization(results, save_path='gold_analysis.png')
```

### 🧠 Machine Learning Enhanced Analysis
```python
from ml_ou_analyzer import MLOUAnalyzer

# Initialize ML analyzer with goal validation
ml_analyzer = MLOUAnalyzer(
    enable_goal_validation=True,  # Enable goal validation
    hidden_sizes=[128, 64, 32],   # Neural network architecture
    num_epochs=100                # Training epochs
)

# Run ML analysis with automatic parameter optimization
results = ml_analyzer.analyze_commodity_ml(
    symbol='GC=F',
    period='5y',
    n_samples=1000,      # Training samples
    n_iterations=1000,   # Optimization iterations
    validate_goal=True   # Validate optimization goals
)

# Create ML-enhanced visualization
fig = ml_analyzer.create_ml_visualization(results, save_path='gold_ml_analysis.png')

# Check goal validation results
if results['goal_validation']:
    print(f"Best alternative objective: {results['goal_validation']['best_alternative_objective']}")
    print(f"Recommendation: {results['goal_validation']['recommendation']}")
```

### 🎯 Goal Validation Example
```python
# Validate that your optimization goal is optimal
validation_results = ml_analyzer.validate_optimization_goal('GC=F', period='5y')

# Create goal validation visualization
fig = ml_analyzer.create_goal_validation_visualization(
    validation_results, 
    save_path='goal_validation_gold.png'
)

# Check statistical significance
if validation_results['statistical_significance']:
    sig_test = validation_results['statistical_significance']
    print(f"Statistical significance: {sig_test['significant']}")
    print(f"P-value: {sig_test['p_value']:.4f}")
    print(f"Effect size: {sig_test['effect_size']}")
```

### Advanced Customization
```python
# Custom OU parameters for specific commodity
analyzer = GenericOUAnalyzer(
    target_mean=1800.0,      # Target gold price
    annual_theta=3.0,        # Faster mean reversion
    corr_length=150,         # Longer correlation window
    threshold=2.5,           # Wider trading bands
    decay_factor=0.98        # Higher EWMA weight
)

# Multi-commodity analysis
commodities = ['GC=F', 'SI=F', 'CL=F', 'NG=F']
results = analyzer.run_analysis(commodities, period='3y', save_plots=True)
```

### Parameter Optimization (Traditional)
```python
# Test different parameter sets
param_sets = [
    {'annual_theta': 2.0, 'threshold': 1.5},
    {'annual_theta': 2.5, 'threshold': 2.0},
    {'annual_theta': 3.0, 'threshold': 2.5}
]

for params in param_sets:
    analyzer = GenericOUAnalyzer(**params)
    results = analyzer.analyze_commodity('GC=F')
    print(f"OU Strength: {results['statistics']['mean_reversion_strength']:.3f}")
```

### 🚀 ML Batch Analysis
```python
# Run ML analysis on multiple commodities
symbols = ['GC=F', 'CL=F', 'NG=F', 'SI=F']  # Gold, Oil, Gas, Silver
results = ml_analyzer.run_ml_analysis(
    symbols=symbols,
    period='5y',
    n_samples=1000,
    n_iterations=1000,
    save_plots=True,
    validate_goals=True  # Enable goal validation for all
)

# Print summary with goal validation results
for symbol, result in results.items():
    print(f"\n{symbol} Analysis:")
    print(f"ML Score: {result['optimization_score']:.4f}")
    if result['goal_validation']:
        goal_val = result['goal_validation']
        print(f"Best Alternative: {goal_val['best_alternative_objective']}")
        print(f"Recommendation: {goal_val['recommendation']}")
```

### 🎯 Quick Start Examples
```bash
# Run traditional OU analysis
python calibrated_ou_analyzer.py

# Run ML-enhanced analysis
python ml_ou_analyzer.py

# Run goal validation example
python goal_validation_example.py

# Run simple ML example
python ml_example.py
```

## 📊 Commodity Universe

### Supported Commodities

| Category | Symbol | Name | Description |
|----------|--------|------|-------------|
| **Energy** | `CL=F` | WTI Crude Oil | West Texas Intermediate |
| | `NG=F` | Natural Gas | Henry Hub Natural Gas |
| **Metals** | `GC=F` | Gold | COMEX Gold Futures |
| | `SI=F` | Silver | COMEX Silver Futures |
| | `HG=F` | Copper | COMEX Copper Futures |
| **Agriculture** | `ZC=F` | Corn | CBOT Corn Futures |
| | `ZW=F` | Wheat | CBOT Wheat Futures |
| | `SB=F` | Sugar | ICE Sugar #11 |

### Data Sources
- **Primary**: Yahoo Finance (yfinance)
- **Frequency**: Daily OHLC data
- **History**: Up to 5 years of historical data
- **Real-time**: Live price updates available

## 📈 Model Performance & Results

### Traditional OU Analysis Results

Recent analysis results across commodity sectors:

| Commodity | Total Return | Volatility | Sharpe Ratio | OU Strength | Half-life |
|-----------|--------------|------------|--------------|-------------|-----------|
| **Gold** | 88.71% | 16.35% | 2.286 | 2.387 | 69 days |
| **Silver** | 80.40% | 29.30% | 1.171 | 1.694 | 97 days |
| **Sugar** | -42.29% | 28.34% | -0.847 | 1.209 | 115 days |
| **Crude Oil** | -30.57% | 31.84% | -0.524 | 0.941 | 147 days |

### 🧠 ML-Enhanced Analysis Results

ML optimization results with goal validation:

| Commodity | ML Score | Optimal θ | Optimal Threshold | Best Objective | Improvement |
|-----------|----------|-----------|-------------------|----------------|-------------|
| **Gold** | 0.4521 | 2.847 | 2.134 | vol_adj_mean_reversion | +12.3% |
| **Silver** | 0.3892 | 3.124 | 1.987 | sharpe_ratio | +8.7% |
| **Crude Oil** | 0.3245 | 2.156 | 2.456 | band_hit_optimization | +15.2% |
| **Natural Gas** | 0.2876 | 4.123 | 1.789 | consistency | +22.1% |

### Model Validation

#### Traditional OU Validation
* **Mean Reversion Strength**: >1.0 indicates strong mean reversion
* **Half-life Range**: 50-200 days optimal for trading strategies
* **Sharpe Ratio**: Risk-adjusted performance metrics
* **Parameter Stability**: Robust across different market regimes

#### ML Validation
* **ML Optimization Score**: Neural network confidence in parameter selection
* **Goal Validation**: Statistical significance of alternative objectives
* **Cross-validation**: Model generalization across different market conditions
* **Backtest Performance**: Real trading strategy validation

## 🎯 Trading Applications

### Mean Reversion Strategy

1. **Entry Signals**:
   - Buy when price crosses below lower OU band
   - Sell when price crosses above upper OU band

2. **Exit Rules**:
   - Take profit when price returns to mean
   - Stop loss based on consecutive bars threshold

3. **Position Sizing**:
   - Size based on mean reversion strength
   - Adjust for volatility and half-life

### Risk Management

* **Dynamic Hedging**: Adjust hedge ratios based on OU parameters
* **Portfolio Construction**: Weight commodities by mean reversion strength
* **Volatility Targeting**: Use OU volatility for position sizing
* **Regime Detection**: Monitor parameter changes for regime shifts

## 🔮 Advanced Features

### Traditional OU Features

#### Customizable Parameters
1. **Target Mean**: Specify fundamental value anchor
2. **Annual Theta**: Control mean reversion speed
3. **Correlation Length**: Adjust parameter smoothing
4. **Threshold**: Set trading band width
5. **Decay Factor**: Control EWMA responsiveness

#### Professional Visualizations
* **Multi-panel Charts**: Price, deviation, consecutive bars, and parameters
* **Statistical Overlays**: OU bands, target mean, half-life indicators
* **Color-coded Analysis**: Visual mean reversion strength indicators
* **Publication-ready**: High-resolution PNG output

#### Comprehensive Statistics
* **Performance Metrics**: Returns, volatility, Sharpe ratio
* **OU Diagnostics**: Mean reversion strength, parameter evolution
* **Risk Measures**: Maximum deviation, consecutive bar analysis
* **Model Validation**: Statistical significance tests

### 🧠 Machine Learning Features

#### Neural Network Optimization
* **Automatic Parameter Tuning**: ML finds optimal OU parameters
* **Multi-objective Optimization**: Tests 8 different optimization goals
* **Statistical Validation**: Ensures optimization goals are truly optimal
* **GPU Acceleration**: CUDA support for faster training

#### Goal Validation System
* **Alternative Objective Testing**: Compares 8 different optimization approaches
* **Statistical Significance**: T-tests and effect size analysis
* **Backtesting Integration**: Real trading strategy validation
* **Intelligent Recommendations**: Actionable guidance on objective selection

#### Advanced ML Capabilities
* **Data Caching**: Intelligent caching for faster repeated analysis
* **Model Persistence**: Save and reuse trained models
* **Cross-validation**: Robust model validation across market conditions
* **Hyperparameter Tuning**: Optimize neural network architecture

#### ML-Enhanced Visualizations
* **Optimization Score Charts**: ML confidence in parameter selection
* **Goal Validation Plots**: Comprehensive objective comparison
* **Statistical Test Results**: P-values and significance indicators
* **Backtest Performance**: Real trading strategy results

## ⚠️ Limitations & Considerations

### Model Limitations

1. **Stationarity Assumption**: OU process assumes constant parameters
2. **Market Regime Changes**: Parameters may change during structural breaks
3. **Transaction Costs**: Not included in signal generation
4. **Liquidity Constraints**: Assumes sufficient market depth

### Data Considerations

1. **Data Quality**: Dependent on yfinance data accuracy
2. **Survivorship Bias**: Only includes currently traded commodities
3. **Rolling Contracts**: Futures contract rollover effects
4. **Market Hours**: 24/7 vs. traditional market hours

## 🚀 Future Enhancements

### Planned Features

1. **Multivariate OU Models**: Cross-commodity mean reversion
2. **Regime-Switching**: Dynamic parameter estimation
3. **High-Frequency Data**: Intraday mean reversion analysis
4. **Advanced ML Models**: LSTM/Transformer-based parameter prediction
5. **Real-time Processing**: Live data feeds and alerts
6. **Ensemble Methods**: Combine multiple ML models for robust optimization

### Research Extensions

1. **Seasonal Adjustments**: Agricultural commodity seasonality
2. **Storage Cost Models**: Physical storage arbitrage
3. **Weather Integration**: Climate impact on agricultural commodities
4. **Macro Factors**: Economic indicator integration
5. **Alternative Data**: Satellite data, sentiment analysis integration
6. **Reinforcement Learning**: Adaptive trading strategy optimization

### ML Enhancements

1. **Deep Reinforcement Learning**: Adaptive parameter optimization
2. **Transfer Learning**: Apply models across different commodities
3. **Online Learning**: Continuous model updates with new data
4. **Explainable AI**: Interpretable ML model decisions
5. **Federated Learning**: Collaborative model training across institutions

## 🤝 Contributing

We welcome contributions from the quantitative finance community! Please feel free to submit a Pull Request.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines

* Follow PEP 8 style guidelines
* Add comprehensive tests for new functionality
* Update documentation for any new features
* Ensure all tests pass before submitting
* Include example usage in docstrings

## 📄 License

This project is licensed under the GNU General Public License - see the LICENSE file for details.

## 🙏 Acknowledgments

* **Author**: YavuzAkbay - For developing this comprehensive OU analysis framework
* **Academic Community**: For foundational work on mean reversion and OU processes
* **Open Source Community**: For the excellent Python libraries that make this project possible
* **Commodity Trading Community**: For providing real-world applications and feedback

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If you get "externally-managed-environment" error
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Data Fetching Issues
```python
# Check internet connection and yfinance status
import yfinance as yf
ticker = yf.Ticker('GC=F')
data = ticker.history(period='1mo')
print(f"Data points: {len(data)}")
```

#### Performance Issues
```python
# Reduce analysis period for faster execution
analyzer = GenericOUAnalyzer()
results = analyzer.analyze_commodity('GC=F', period='1y')  # Instead of '5y'
```

### Getting Help

1. **Check the examples**: Run the analysis with default parameters
2. **Verify dependencies**: Ensure all packages are installed correctly
3. **Check the logs**: Look for error messages in the console output
4. **Data validation**: Verify commodity symbols are correct

## 📞 Contact & Support

* **Email**: akbay.yavuz@gmail.com
* **LinkedIn**: <https://www.linkedin.com/in/yavuzakbay/>
* **GitHub Issues**: Create an issue for bug reports or feature requests

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

_Built with ❤️ by YavuzAkbay for the quantitative finance and commodity trading community_

## 📚 References

* Vasicek, O. (1977). An equilibrium characterization of the term structure. _Journal of Financial Economics_
* Ornstein, L. S., & Uhlenbeck, G. E. (1930). On the theory of the Brownian motion. _Physical Review_
* Schwartz, E. S. (1997). The stochastic behavior of commodity prices: Implications for valuation and hedging. _Journal of Finance_
* Pindyck, R. S. (2001). The dynamics of commodity spot and futures markets: A primer. _Energy Journal_

## About

A comprehensive Ornstein-Uhlenbeck process analyzer for commodity mean reversion analysis, combining rigorous mathematical foundations with practical trading applications.

### Resources

* **Documentation**: Comprehensive usage examples and API reference
* **Examples**: Ready-to-run analysis scripts
* **Visualizations**: Professional charts and statistical overlays

### License

GPL-3.0 license

### Languages

* Python 100.0%
