#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import logging
import pickle
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class OUDataCache:
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_path(self, symbol: str, period: str) -> Path:
        safe_symbol = symbol.replace('=', '_').replace('^', '_')
        return self.cache_dir / f"{safe_symbol}_{period}.pkl"
    
    def load_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        cache_path = self.get_cache_path(symbol, period)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded cached data for {symbol} ({period})")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
        return None
    
    def save_data(self, symbol: str, period: str, data: pd.DataFrame):
        cache_path = self.get_cache_path(symbol, period)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cached data for {symbol} ({period})")
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol}: {e}")


class OUParameterDataset(Dataset):
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class OUParameterNet(nn.Module):
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], 
                 output_size: int = 1, dropout_rate: float = 0.2):
        super(OUParameterNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class GoalValidator:
    
    def __init__(self):
        self.validation_results = {}
        
    def calculate_alternative_objectives(self, 
                                      prices: pd.Series,
                                      returns: pd.Series,
                                      deviation: pd.Series,
                                      metrics: Dict[str, float]) -> Dict[str, float]:
        objectives = {}
        
        objectives['pure_mean_reversion'] = metrics['mean_reversion_strength']
        objectives['sharpe_ratio'] = metrics['sharpe_ratio']
        
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        objectives['max_drawdown'] = -drawdown.min()
        
        vol_adj_mean_rev = metrics['mean_reversion_strength'] / (metrics['volatility_annualized'] / 100)
        objectives['vol_adj_mean_reversion'] = vol_adj_mean_rev
        
        optimal_hit_rate = 0.15
        hit_rate_penalty = abs(metrics['band_hit_rate'] - optimal_hit_rate)
        objectives['band_hit_optimization'] = 1 - hit_rate_penalty
        
        rolling_returns = returns.rolling(window=20).mean()
        consistency = 1 / (rolling_returns.std() + 1e-6)
        objectives['consistency'] = consistency
        
        benchmark_return = returns.mean()
        excess_returns = returns - benchmark_return
        tracking_error = excess_returns.std()
        information_ratio = excess_returns.mean() / (tracking_error + 1e-6)
        objectives['information_ratio'] = information_ratio
        
        calmar_ratio = metrics['total_return'] / (abs(objectives['max_drawdown']) + 1e-6)
        objectives['calmar_ratio'] = calmar_ratio
        
        return objectives
    
    def backtest_strategy(self, 
                         prices: pd.Series,
                         upper_band: pd.Series,
                         lower_band: pd.Series,
                         mean: pd.Series) -> Dict[str, float]:
        signals = pd.Series(0, index=prices.index)
        positions = pd.Series(0, index=prices.index)
        
        for i in range(1, len(prices)):
            if prices.iloc[i] > upper_band.iloc[i]:
                signals.iloc[i] = -1
            elif prices.iloc[i] < lower_band.iloc[i]:
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
                
            positions.iloc[i] = signals.iloc[i]
        
        strategy_returns = positions.shift(1) * prices.pct_change()
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {'strategy_return': 0, 'strategy_sharpe': 0, 'win_rate': 0}
        
        strategy_return = (1 + strategy_returns).prod() - 1
        strategy_sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-6) * np.sqrt(252)
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'strategy_return': strategy_return * 100,
            'strategy_sharpe': strategy_sharpe,
            'win_rate': win_rate
        }
    
    def statistical_significance_test(self, 
                                    optimization_scores: List[float],
                                    baseline_scores: List[float]) -> Dict[str, float]:
        from scipy import stats
        
        t_stat, p_value = stats.ttest_ind(optimization_scores, baseline_scores)
        
        pooled_std = np.sqrt(((len(optimization_scores) - 1) * np.var(optimization_scores, ddof=1) + 
                             (len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1)) / 
                            (len(optimization_scores) + len(baseline_scores) - 2))
        cohens_d = (np.mean(optimization_scores) - np.mean(baseline_scores)) / pooled_std
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    
    def cross_validate_objectives(self, 
                                symbol: str,
                                data: pd.DataFrame,
                                param_combinations: List[Dict],
                                objective_functions: List[str]) -> Dict[str, Dict]:
        results = {}
        
        for obj_func in objective_functions:
            scores = []
            
            for params in param_combinations[:100]:
                try:
                    prices = data['close']
                    returns = np.log(prices / prices.shift(1)).dropna()
                    
                    ewma = prices.ewm(alpha=1-params['decay_factor'], adjust=False).mean()
                    volatility = prices.rolling(window=params['corr_length']).std()
                    upper_band = ewma + params['threshold'] * volatility
                    lower_band = ewma - params['threshold'] * volatility
                    deviation = (prices - ewma) / volatility
                    
                    autocorr = returns.rolling(window=params['corr_length']).apply(
                        lambda x: x.autocorr(lag=1) if len(x) == params['corr_length'] else np.nan
                    ).dropna()
                    
                    mean_reversion_strength = abs(autocorr.mean()) if len(autocorr) > 0 else 0
                    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
                    volatility_annualized = returns.std() * np.sqrt(252) * 100
                    sharpe_ratio = total_return / volatility_annualized if volatility_annualized > 0 else 0
                    
                    metrics = {
                        'mean_reversion_strength': mean_reversion_strength,
                        'total_return': total_return,
                        'volatility_annualized': volatility_annualized,
                        'sharpe_ratio': sharpe_ratio
                    }
                    
                    alt_objectives = self.calculate_alternative_objectives(
                        prices, returns, deviation, metrics
                    )
                    
                    if obj_func in alt_objectives:
                        scores.append(alt_objectives[obj_func])
                    else:
                        scores.append(0)
                        
                except Exception as e:
                    continue
            
            if scores:
                results[obj_func] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'max_score': np.max(scores),
                    'min_score': np.min(scores),
                    'scores': scores
                }
        
        return results


class MLOUAnalyzer:
    
    def __init__(self, 
                 cache_dir: str = "data_cache",
                 model_dir: str = "models",
                 hidden_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 enable_goal_validation: bool = True):
        self.cache_dir = cache_dir
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.enable_goal_validation = enable_goal_validation
        
        self.data_cache = OUDataCache(cache_dir)
        self.scaler = StandardScaler()
        self.models = {}
        self.optimization_results = {}
        
        if self.enable_goal_validation:
            self.goal_validator = GoalValidator()
        
        self.param_ranges = {
            'annual_theta': (0.5, 5.0),
            'corr_length': (20, 200),
            'threshold': (1.0, 3.0),
            'decay_factor': (0.85, 0.99)
        }
        
        self.alternative_objectives = [
            'pure_mean_reversion',
            'sharpe_ratio', 
            'vol_adj_mean_reversion',
            'band_hit_optimization',
            'consistency',
            'information_ratio',
            'calmar_ratio'
        ]
        
    def fetch_commodity_data(self, 
                           symbol: str, 
                           period: str = "5y",
                           interval: str = "1d",
                           use_cache: bool = True) -> pd.DataFrame:
        if use_cache:
            cached_data = self.data_cache.load_data(symbol, period)
            if cached_data is not None:
                return cached_data
        
        logger.info(f"Fetching data for {symbol} over {period} period...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            data = data.dropna()
            data.columns = [col.lower() for col in data.columns]
            
            if use_cache:
                self.data_cache.save_data(symbol, period, data)
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def calculate_ou_metrics(self, 
                           prices: pd.Series,
                           annual_theta: float,
                           corr_length: int,
                           threshold: float,
                           decay_factor: float,
                           target_mean: Optional[float] = None) -> Dict[str, float]:
        returns = np.log(prices / prices.shift(1)).dropna()
        
        ewma = prices.ewm(alpha=1-decay_factor, adjust=False).mean()
        if target_mean is not None:
            mean = 0.7 * ewma + 0.3 * target_mean
        else:
            mean = ewma
        
        volatility = prices.rolling(window=corr_length).std()
        deviation = (prices - mean) / volatility
        upper_band = mean + threshold * volatility
        lower_band = mean - threshold * volatility
        
        metrics = {}
        
        autocorr = returns.rolling(window=corr_length).apply(
            lambda x: x.autocorr(lag=1) if len(x) == corr_length else np.nan
        ).dropna()
        
        if len(autocorr) > 0:
            mean_autocorr = autocorr.mean()
            metrics['mean_reversion_strength'] = abs(mean_autocorr)
        else:
            metrics['mean_reversion_strength'] = 0.0
        
        deviation_clean = deviation.dropna()
        if len(deviation_clean) > 0:
            metrics['mean_deviation'] = deviation_clean.mean()
            metrics['std_deviation'] = deviation_clean.std()
            metrics['max_deviation'] = deviation_clean.max()
            metrics['min_deviation'] = deviation_clean.min()
        else:
            metrics['mean_deviation'] = 0.0
            metrics['std_deviation'] = 1.0
            metrics['max_deviation'] = 0.0
            metrics['min_deviation'] = 0.0
        
        above_upper = (prices > upper_band).sum()
        below_lower = (prices < lower_band).sum()
        total_points = len(prices.dropna())
        metrics['band_hit_rate'] = (above_upper + below_lower) / total_points if total_points > 0 else 0.0
        
        metrics['total_return'] = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        metrics['volatility_annualized'] = returns.std() * np.sqrt(252) * 100
        metrics['sharpe_ratio'] = metrics['total_return'] / metrics['volatility_annualized'] if metrics['volatility_annualized'] > 0 else 0
        
        daily_theta = annual_theta / 252
        metrics['half_life_days'] = int(np.log(2) / daily_theta) if daily_theta > 0 else 1000
        
        return metrics
    
    def generate_parameter_combinations(self, n_samples: int = 1000) -> List[Dict]:
        combinations = []
        
        for _ in range(n_samples):
            params = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                if param == 'corr_length':
                    params[param] = int(np.random.uniform(min_val, max_val))
                else:
                    params[param] = np.random.uniform(min_val, max_val)
            combinations.append(params)
        
        return combinations
    
    def validate_optimization_goal(self, 
                                 symbol: str,
                                 period: str = "5y",
                                 n_samples: int = 500) -> Dict:
        if not self.enable_goal_validation:
            return {'validation_enabled': False}
        
        logger.info(f"Validating optimization goal for {symbol}...")
        
        data = self.fetch_commodity_data(symbol, period)
        prices = data['close']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        param_combinations = self.generate_parameter_combinations(n_samples)
        
        objective_results = self.goal_validator.cross_validate_objectives(
            symbol, data, param_combinations, self.alternative_objectives
        )
        
        current_scores = []
        for params in param_combinations[:100]:
            try:
                metrics = self.calculate_ou_metrics(
                    prices=prices,
                    annual_theta=params['annual_theta'],
                    corr_length=params['corr_length'],
                    threshold=params['threshold'],
                    decay_factor=params['decay_factor']
                )
                current_score = metrics['mean_reversion_strength'] * 0.7 + (metrics['sharpe_ratio'] / 10) * 0.3
                current_scores.append(current_score)
            except:
                continue
        
        best_objective = None
        best_score = -float('inf')
        
        for obj_name, obj_results in objective_results.items():
            if obj_results['mean_score'] > best_score:
                best_score = obj_results['mean_score']
                best_objective = obj_name
        
        significance_test = None
        if current_scores and best_objective in objective_results:
            significance_test = self.goal_validator.statistical_significance_test(
                objective_results[best_objective]['scores'],
                current_scores
            )
        
        backtest_results = {}
        for obj_name in ['current', best_objective]:
            if obj_name == 'current':
                params = {
                    'annual_theta': 2.5,
                    'corr_length': 100,
                    'threshold': 2.0,
                    'decay_factor': 0.95
                }
            else:
                best_params = None
                best_score = -float('inf')
                for p in param_combinations[:50]:
                    try:
                        metrics = self.calculate_ou_metrics(
                            prices=prices,
                            annual_theta=p['annual_theta'],
                            corr_length=p['corr_length'],
                            threshold=p['threshold'],
                            decay_factor=p['decay_factor']
                        )
                        
                        ewma = prices.ewm(alpha=1-p['decay_factor'], adjust=False).mean()
                        volatility = prices.rolling(window=p['corr_length']).std()
                        upper_band = ewma + p['threshold'] * volatility
                        lower_band = ewma - p['threshold'] * volatility
                        deviation = (prices - ewma) / volatility
                        
                        alt_objectives = self.goal_validator.calculate_alternative_objectives(
                            prices, returns, deviation, metrics
                        )
                        
                        if obj_name in alt_objectives and alt_objectives[obj_name] > best_score:
                            best_score = alt_objectives[obj_name]
                            best_params = p
                    except:
                        continue
                
                if best_params:
                    params = best_params
                else:
                    continue
            
            ewma = prices.ewm(alpha=1-params['decay_factor'], adjust=False).mean()
            volatility = prices.rolling(window=params['corr_length']).std()
            upper_band = ewma + params['threshold'] * volatility
            lower_band = ewma - params['threshold'] * volatility
            
            backtest_results[obj_name] = self.goal_validator.backtest_strategy(
                prices, upper_band, lower_band, ewma
            )
        
        validation_result = {
            'symbol': symbol,
            'current_objective_scores': {
                'mean': np.mean(current_scores) if current_scores else 0,
                'std': np.std(current_scores) if current_scores else 0,
                'scores': current_scores
            },
            'alternative_objectives': objective_results,
            'best_alternative_objective': best_objective,
            'best_alternative_score': best_score,
            'statistical_significance': significance_test,
            'backtest_comparison': backtest_results,
            'recommendation': self._generate_goal_recommendation(
                current_scores, objective_results, best_objective, significance_test
            )
        }
        
        self.goal_validator.validation_results[symbol] = validation_result
        logger.info(f"Goal validation completed for {symbol}")
        
        return validation_result
    
    def _generate_goal_recommendation(self, 
                                    current_scores: List[float],
                                    objective_results: Dict,
                                    best_objective: str,
                                    significance_test: Optional[Dict]) -> str:
        if not current_scores or not objective_results:
            return "Insufficient data for validation"
        
        current_mean = np.mean(current_scores)
        best_alternative_score = objective_results[best_objective]['mean_score']
        
        improvement = (best_alternative_score - current_mean) / current_mean * 100
        
        if significance_test and significance_test['significant']:
            if improvement > 10:
                return f"STRONG RECOMMENDATION: Switch to '{best_objective}' objective. {improvement:.1f}% improvement with statistical significance (p={significance_test['p_value']:.4f})"
            elif improvement > 5:
                return f"MODERATE RECOMMENDATION: Consider '{best_objective}' objective. {improvement:.1f}% improvement with statistical significance (p={significance_test['p_value']:.4f})"
            else:
                return f"WEAK RECOMMENDATION: '{best_objective}' shows {improvement:.1f}% improvement but may not be practically significant"
        else:
            if improvement > 15:
                return f"EXPLORATORY RECOMMENDATION: '{best_objective}' shows {improvement:.1f}% improvement but lacks statistical significance. Consider larger sample size."
            else:
                return f"NO CHANGE RECOMMENDED: Current objective performs adequately. '{best_objective}' shows {improvement:.1f}% improvement but not statistically significant."
    
    def create_training_data(self, 
                           symbol: str,
                           period: str = "5y",
                           n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Creating training data for {symbol}...")
        
        data = self.fetch_commodity_data(symbol, period)
        prices = data['close']
        
        param_combinations = self.generate_parameter_combinations(n_samples)
        
        features = []
        targets = []
        
        for params in param_combinations:
            try:
                metrics = self.calculate_ou_metrics(
                    prices=prices,
                    annual_theta=params['annual_theta'],
                    corr_length=params['corr_length'],
                    threshold=params['threshold'],
                    decay_factor=params['decay_factor']
                )
                
                feature_vector = [
                    prices.mean(),
                    prices.std(),
                    prices.skew(),
                    prices.kurtosis(),
                    (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                    params['annual_theta'],
                    params['corr_length'],
                    params['threshold'],
                    params['decay_factor']
                ]
                
                target = metrics['mean_reversion_strength'] * 0.7 + (metrics['sharpe_ratio'] / 10) * 0.3
                
                features.append(feature_vector)
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for parameters {params}: {e}")
                continue
        
        features = np.array(features)
        targets = np.array(targets)
        
        logger.info(f"Created training data: {len(features)} samples")
        return features, targets
    
    def train_parameter_model(self, 
                            symbol: str,
                            period: str = "5y",
                            n_samples: int = 1000) -> nn.Module:
        logger.info(f"Training parameter optimization model for {symbol}...")
        
        features, targets = self.create_training_data(symbol, period, n_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        train_dataset = OUParameterDataset(X_train_scaled, y_train)
        test_dataset = OUParameterDataset(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        input_size = X_train_scaled.shape[1]
        model = OUParameterNet(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            dropout_rate=self.dropout_rate
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_targets)
                    test_loss += loss.item()
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
        
        model_path = self.model_dir / f"{symbol.replace('=', '_').replace('^', '_')}_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': self.scaler,
            'train_losses': train_losses,
            'test_losses': test_losses
        }, model_path)
        
        self.models[symbol] = model
        logger.info(f"Model trained and saved for {symbol}")
        
        return model
    
    def optimize_parameters(self, 
                          symbol: str,
                          period: str = "5y",
                          n_iterations: int = 1000) -> Dict:
        logger.info(f"Optimizing parameters for {symbol}...")
        
        if symbol not in self.models:
            self.train_parameter_model(symbol, period)
        
        model = self.models[symbol]
        model.eval()
        
        candidates = self.generate_parameter_combinations(n_iterations)
        
        best_score = -float('inf')
        best_params = None
        
        data = self.fetch_commodity_data(symbol, period)
        prices = data['close']
        
        price_features = [
            prices.mean(),
            prices.std(),
            prices.skew(),
            prices.kurtosis(),
            (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        ]
        
        with torch.no_grad():
            for params in candidates:
                feature_vector = price_features + [
                    params['annual_theta'],
                    params['corr_length'],
                    params['threshold'],
                    params['decay_factor']
                ]
                
                feature_tensor = torch.FloatTensor(
                    self.scaler.transform([feature_vector])
                ).to(device)
                
                score = model(feature_tensor).item()
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        
        best_metrics = self.calculate_ou_metrics(
            prices=prices,
            annual_theta=best_params['annual_theta'],
            corr_length=best_params['corr_length'],
            threshold=best_params['threshold'],
            decay_factor=best_params['decay_factor']
        )
        
        result = {
            'symbol': symbol,
            'optimal_parameters': best_params,
            'optimization_score': best_score,
            'metrics': best_metrics,
            'period': period
        }
        
        self.optimization_results[symbol] = result
        logger.info(f"Parameter optimization completed for {symbol}")
        
        return result
    
    def analyze_commodity_ml(self, 
                           symbol: str,
                           period: str = "5y",
                           n_samples: int = 1000,
                           n_iterations: int = 1000,
                           validate_goal: bool = True) -> Dict:
        logger.info(f"Starting ML OU analysis for {symbol}...")
        
        goal_validation = None
        if validate_goal and self.enable_goal_validation:
            goal_validation = self.validate_optimization_goal(symbol, period, n_samples//2)
        
        optimization_result = self.optimize_parameters(symbol, period, n_iterations)
        
        optimal_params = optimization_result['optimal_parameters']
        
        data = self.fetch_commodity_data(symbol, period)
        prices = data['close']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        ewma = prices.ewm(alpha=1-optimal_params['decay_factor'], adjust=False).mean()
        mean = ewma
        volatility = prices.rolling(window=optimal_params['corr_length']).std()
        upper_band = mean + optimal_params['threshold'] * volatility
        lower_band = mean - optimal_params['threshold'] * volatility
        deviation = (prices - mean) / volatility
        
        consecutive_bars = pd.Series(0, index=deviation.index)
        for i in range(1, len(deviation)):
            if ((prices.iloc[i-1] < mean.iloc[i-1] and prices.iloc[i] >= mean.iloc[i]) or
                (prices.iloc[i-1] > mean.iloc[i-1] and prices.iloc[i] <= mean.iloc[i])):
                consecutive_bars.iloc[i] = 0
            elif abs(deviation.iloc[i]) > 0:
                consecutive_bars.iloc[i] = min(consecutive_bars.iloc[i-1] + 1, 400)
        results = {
            'symbol': symbol,
            'data': data,
            'prices': prices,
            'returns': returns,
            'mean': mean,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'deviation': deviation,
            'consecutive_bars': consecutive_bars,
            'optimal_parameters': optimal_params,
            'optimization_score': optimization_result['optimization_score'],
            'metrics': optimization_result['metrics'],
            'period': period,
            'goal_validation': goal_validation
        }
        
        logger.info(f"ML OU analysis completed for {symbol}")
        return results
    
    def create_ml_visualization(self, 
                              results: Dict, 
                              save_path: str = None,
                              figsize: Tuple[int, int] = (18, 14)) -> plt.Figure:
        symbol = results['symbol']
        prices = results['prices']
        mean = results['mean']
        upper_band = results['upper_band']
        lower_band = results['lower_band']
        deviation = results['deviation']
        consecutive_bars = results['consecutive_bars']
        optimal_params = results['optimal_parameters']
        metrics = results['metrics']
        optimization_score = results['optimization_score']
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(prices.index, prices, 'k-', linewidth=1.5, label='Price', alpha=0.8)
        ax1.plot(mean.index, mean, 'b-', linewidth=2, label='ML-Optimized Mean (μ)', alpha=0.9)
        ax1.plot(upper_band.index, upper_band, 'g--', linewidth=1.5, 
                label=f'Upper Band (+{optimal_params["threshold"]:.1f}σ)', alpha=0.7)
        ax1.plot(lower_band.index, lower_band, 'r--', linewidth=1.5, 
                label=f'Lower Band (-{optimal_params["threshold"]:.1f}σ)', alpha=0.7)
        
        ax1.fill_between(mean.index, mean, upper_band, 
                        where=(deviation > 0), 
                        color='green', alpha=0.1, interpolate=True)
        ax1.fill_between(mean.index, mean, lower_band, 
                        where=(deviation < 0), 
                        color='red', alpha=0.1, interpolate=True)
        
        ax1.set_title(f'ML-Enhanced OU Analysis: {symbol}', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        colors = ['red' if d < 0 else 'green' for d in deviation]
        ax2.bar(deviation.index, deviation, color=colors, alpha=0.6, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=optimal_params['threshold'], color='green', linestyle='--', alpha=0.7)
        ax2.axhline(y=-optimal_params['threshold'], color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Deviation (σ)', fontsize=12)
        ax2.set_title('Standardized Deviation from ML-Optimized Mean', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax3.plot(consecutive_bars.index, consecutive_bars, 'purple', linewidth=1.5)
        ax3.axhline(y=metrics['half_life_days'], color='red', linestyle='--', alpha=0.7, 
                   label=f'ML-Optimized Half-life ({metrics["half_life_days"]} days)')
        ax3.set_ylabel('Consecutive Bars', fontsize=12)
        ax3.set_title('Consecutive Bars Away from Mean', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        ax4.axhline(y=optimization_score, color='orange', linewidth=3, 
                   label=f'ML Optimization Score: {optimization_score:.4f}')
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_title('ML Parameter Optimization Score', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[:, 1])
        ax5.axis('off')
        ml_text = f"""
        ML-ENHANCED OU ANALYSIS
        ======================
        
        Symbol: {symbol}
        Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}
        Data Points: {len(prices):,}
        
        ML-OPTIMIZED PARAMETERS
        -----------------------
        Annual θ: {optimal_params['annual_theta']:.3f}
        Correlation Length: {optimal_params['corr_length']} days
        Threshold: {optimal_params['threshold']:.3f}σ
        Decay Factor: {optimal_params['decay_factor']:.4f}
        Half-life: {metrics['half_life_days']} days
        
        OPTIMIZATION RESULTS
        --------------------
        ML Score: {optimization_score:.6f}
        Mean Reversion Strength: {metrics['mean_reversion_strength']:.4f}
        Band Hit Rate: {metrics['band_hit_rate']:.2%}
        
        PERFORMANCE METRICS
        -------------------
        Total Return: {metrics['total_return']:.2f}%
        Annualized Volatility: {metrics['volatility_annualized']:.2f}%
        Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        
        DEVIATION STATISTICS
        --------------------
        Mean Deviation: {metrics['mean_deviation']:.4f}
        Std Deviation: {metrics['std_deviation']:.4f}
        Max Deviation: {metrics['max_deviation']:.4f}
        Min Deviation: {metrics['min_deviation']:.4f}
        
        ML INTERPRETATION
        ------------------
        """
        
        if optimization_score > 0.5:
            ml_text += "Excellent ML optimization\n"
        elif optimization_score > 0.3:
            ml_text += "Good ML optimization\n"
        else:
            ml_text += "Moderate ML optimization\n"
        
        if metrics['mean_reversion_strength'] > 0.3:
            ml_text += "Strong mean reversion detected\n"
        elif metrics['mean_reversion_strength'] > 0.1:
            ml_text += "Moderate mean reversion\n"
        else:
            ml_text += "Weak mean reversion\n"
        
        if metrics['band_hit_rate'] > 0.1:
            ml_text += "Frequent band crossings\n"
        else:
            ml_text += "Infrequent band crossings\n"
        
        ax5.text(0.05, 0.95, ml_text, transform=ax5.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ML visualization saved to {save_path}")
        
        return fig
    
    def create_goal_validation_visualization(self, 
                                           validation_results: Dict,
                                           save_path: str = None,
                                           figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        symbol = validation_results['symbol']
        current_scores = validation_results['current_objective_scores']
        alternative_objectives = validation_results['alternative_objectives']
        best_objective = validation_results['best_alternative_objective']
        significance_test = validation_results['statistical_significance']
        backtest_comparison = validation_results['backtest_comparison']
        recommendation = validation_results['recommendation']
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        
        objectives = list(alternative_objectives.keys())
        means = [alternative_objectives[obj]['mean_score'] for obj in objectives]
        stds = [alternative_objectives[obj]['std_score'] for obj in objectives]
        
        objectives.insert(0, 'Current (MR+Sharpe)')
        means.insert(0, current_scores['mean'])
        stds.insert(0, current_scores['std'])
        
        colors = ['red' if obj == 'Current (MR+Sharpe)' else 'blue' if obj == best_objective else 'gray' 
                 for obj in objectives]
        
        bars = ax1.bar(objectives, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax1.set_title('Objective Function Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Score', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        if best_objective in objectives:
            best_idx = objectives.index(best_objective)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        ax2 = fig.add_subplot(gs[1, 0])
        
        if backtest_comparison:
            strategies = list(backtest_comparison.keys())
            returns = [backtest_comparison[s]['strategy_return'] for s in strategies]
            sharpes = [backtest_comparison[s]['strategy_sharpe'] for s in strategies]
            
            x = np.arange(len(strategies))
            width = 0.35
            
            ax2_twin = ax2.twinx()
            bars1 = ax2.bar(x - width/2, returns, width, label='Strategy Return (%)', alpha=0.7)
            bars2 = ax2_twin.bar(x + width/2, sharpes, width, label='Strategy Sharpe', alpha=0.7, color='orange')
            
            ax2.set_xlabel('Strategy')
            ax2.set_ylabel('Strategy Return (%)', color='blue')
            ax2_twin.set_ylabel('Strategy Sharpe', color='orange')
            ax2.set_title('Backtest Performance Comparison', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(strategies)
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[2, 0])
        
        if significance_test:
            metrics = ['T-statistic', 'P-value', 'Cohen\'s d']
            values = [
                abs(significance_test['t_statistic']),
                significance_test['p_value'],
                abs(significance_test['cohens_d'])
            ]
            colors = ['green' if significance_test['significant'] else 'red',
                     'green' if significance_test['p_value'] < 0.05 else 'red',
                     'blue']
            
            bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
            ax3.set_title('Statistical Significance Test', fontsize=12)
            ax3.set_ylabel('Value')
            ax3.grid(True, alpha=0.3)
            
            ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
            ax3.legend()
        
        ax4 = fig.add_subplot(gs[:, 1])
        ax4.axis('off')
        results_text = f"""
        GOAL VALIDATION RESULTS
        =======================
        
        Symbol: {symbol}
        
        CURRENT OBJECTIVE
        -----------------
        Mean Score: {current_scores['mean']:.4f}
        Std Score: {current_scores['std']:.4f}
        
        BEST ALTERNATIVE
        ----------------
        Objective: {best_objective}
        Mean Score: {alternative_objectives[best_objective]['mean_score']:.4f}
        Improvement: {((alternative_objectives[best_objective]['mean_score'] - current_scores['mean']) / current_scores['mean'] * 100):.1f}%
        
        STATISTICAL TEST
        ----------------
        """
        
        if significance_test:
            results_text += f"""
        T-statistic: {significance_test['t_statistic']:.4f}
        P-value: {significance_test['p_value']:.4f}
        Cohen's d: {significance_test['cohens_d']:.4f}
        Significant: {'Yes' if significance_test['significant'] else 'No'}
        Effect Size: {significance_test['effect_size']}
        """
        
        results_text += f"""
        
        BACKTEST RESULTS
        ----------------
        """
        
        if backtest_comparison:
            for strategy, metrics in backtest_comparison.items():
                results_text += f"""
        {strategy.upper()}:
          Return: {metrics['strategy_return']:.2f}%
          Sharpe: {metrics['strategy_sharpe']:.3f}
          Win Rate: {metrics['win_rate']:.2%}
        """
        
        results_text += f"""
        
        RECOMMENDATION
        --------------
        {recommendation}
        """
        
        ax4.text(0.05, 0.95, results_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Goal validation visualization saved to {save_path}")
        
        return fig
    
    def run_ml_analysis(self, 
                       symbols: List[str], 
                       period: str = "5y",
                       n_samples: int = 1000,
                       n_iterations: int = 1000,
                       save_plots: bool = True,
                       validate_goals: bool = True) -> Dict[str, Dict]:
        all_results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Running ML analysis for {symbol}...")
                results = self.analyze_commodity_ml(
                    symbol=symbol,
                    period=period,
                    n_samples=n_samples,
                    n_iterations=n_iterations,
                    validate_goal=validate_goals
                )
                all_results[symbol] = results
                
                if save_plots:
                    filename = f"ml_ou_analysis_{symbol.replace('=', '_').replace('^', '_')}.png"
                    self.create_ml_visualization(results, save_path=filename)
                    
                    if results['goal_validation'] and results['goal_validation'].get('validation_enabled', True):
                        validation_filename = f"goal_validation_{symbol.replace('=', '_').replace('^', '_')}.png"
                        self.create_goal_validation_visualization(
                            results['goal_validation'], save_path=validation_filename
                        )
                optimal_params = results['optimal_parameters']
                metrics = results['metrics']
                print(f"\n{'='*80}")
                print(f"ML OU ANALYSIS: {symbol}")
                print(f"{'='*80}")
                print(f"ML Optimization Score: {results['optimization_score']:.6f}")
                print(f"Optimal Annual θ: {optimal_params['annual_theta']:.3f}")
                print(f"Optimal Correlation Length: {optimal_params['corr_length']} days")
                print(f"Optimal Threshold: {optimal_params['threshold']:.3f}σ")
                print(f"Optimal Decay Factor: {optimal_params['decay_factor']:.4f}")
                print(f"Half-life: {metrics['half_life_days']} days")
                print(f"Mean Reversion Strength: {metrics['mean_reversion_strength']:.4f}")
                print(f"Band Hit Rate: {metrics['band_hit_rate']:.2%}")
                print(f"Total Return: {metrics['total_return']:.2f}%")
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                
                if results['goal_validation'] and results['goal_validation'].get('validation_enabled', True):
                    goal_val = results['goal_validation']
                    print(f"\nGOAL VALIDATION:")
                    print(f"Best Alternative: {goal_val['best_alternative_objective']}")
                    print(f"Improvement: {((goal_val['best_alternative_score'] - goal_val['current_objective_scores']['mean']) / goal_val['current_objective_scores']['mean'] * 100):.1f}%")
                    if goal_val['statistical_significance']:
                        print(f"Statistical Significance: {'Yes' if goal_val['statistical_significance']['significant'] else 'No'} (p={goal_val['statistical_significance']['p_value']:.4f})")
                    print(f"Recommendation: {goal_val['recommendation']}")
                
                print(f"{'='*80}")
                
            except Exception as e:
                logger.error(f"Error in ML analysis for {symbol}: {str(e)}")
                continue
        
        return all_results


def main():
    print("Machine Learning Enhanced Ornstein-Uhlenbeck Process Analyzer")
    print("=" * 70)
    print("PyTorch-based Parameter Optimization for Commodity Analysis")
    print("=" * 70)
    
    ml_analyzer = MLOUAnalyzer(
        cache_dir="data_cache",
        model_dir="models",
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100
    )
    commodities = {
        'Natural Gas': 'NG=F',
        'Crude Oil': 'CL=F',
        'Sugar': 'SB=F',
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Copper': 'HG=F',
        'Wheat': 'ZW=F',
        'Corn': 'ZC=F'
    }
    
    print(f"\nRunning ML analysis on {len(commodities)} commodities...")
    print("Commodities:", list(commodities.keys()))
    print(f"Using device: {device}")
    
    results = ml_analyzer.run_ml_analysis(
        symbols=list(commodities.values()),
        period="2y",
        n_samples=1000,
        n_iterations=1000,
        save_plots=True,
        validate_goals=True
    )
    print(f"\n{'='*120}")
    print("ML ANALYSIS SUMMARY")
    print(f"{'='*120}")
    print(f"{'Commodity':<15} {'ML Score':<10} {'Annual θ':<10} {'Corr Len':<10} {'Threshold':<10} {'Decay':<8} {'Half-life':<10} {'OU Strength':<12} {'Return%':<10} {'Sharpe':<8}")
    print("-" * 120)
    
    for name, symbol in commodities.items():
        if symbol in results:
            optimal_params = results[symbol]['optimal_parameters']
            metrics = results[symbol]['metrics']
            ml_score = results[symbol]['optimization_score']
            
            print(f"{name:<15} {ml_score:<10.4f} {optimal_params['annual_theta']:<10.3f} {optimal_params['corr_length']:<10} "
                  f"{optimal_params['threshold']:<10.3f} {optimal_params['decay_factor']:<8.4f} {metrics['half_life_days']:<10} "
                  f"{metrics['mean_reversion_strength']:<12.4f} {metrics['total_return']:<10.2f} {metrics['sharpe_ratio']:<8.3f}")
    
    print(f"\n{'='*120}")
    print("ML Analysis completed successfully!")
    print("Individual ML-enhanced commodity charts saved as PNG files.")
    print("Trained models saved in 'models' directory.")
    print("Cached data saved in 'data_cache' directory.")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
