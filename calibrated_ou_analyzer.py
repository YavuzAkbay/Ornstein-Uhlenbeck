#!/usr/bin/env python3
"""
Generic Ornstein-Uhlenbeck Process Analyzer for Commodities
==========================================================

A flexible OU process analyzer that allows custom parameter specification
for commodity mean reversion analysis.

Author: Quantitative Analyst
License: Scientific Research Use
"""

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
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for professional appearance
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class GenericOUAnalyzer:
    """
    Generic Ornstein-Uhlenbeck Process Analyzer
    
    This class implements OU process with customizable parameters
    for commodity mean reversion analysis.
    """
    
    def __init__(self, 
                 target_mean: Optional[float] = None,
                 annual_theta: float = 2.5,
                 corr_length: int = 100,
                 threshold: float = 2.0,
                 decay_factor: float = 0.95):
        """
        Initialize with customizable parameters
        
        Parameters:
        -----------
        target_mean : float, optional
            Target long-term mean (if None, will be data-driven)
        annual_theta : float
            Annual mean reversion rate (default: 2.5)
        corr_length : int
            Correlation window length (default: 100)
        threshold : float
            Standard deviation threshold for bands (default: 2.0)
        decay_factor : float
            EWMA decay factor (default: 0.95)
        """
        self.target_mean = target_mean
        self.annual_theta = annual_theta
        self.corr_length = corr_length
        self.threshold = threshold
        self.decay_factor = decay_factor
        
        # Calculate half-life in days
        self.half_life_days = int(np.log(2) / annual_theta * 252)
        
        # Initialize storage for analysis results
        self.data = None
        self.ou_params = {}
        self.statistics = {}
        
    def fetch_commodity_data(self, 
                           symbol: str, 
                           period: str = "5y",
                           interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical commodity data using yfinance
        """
        logger.info(f"Fetching data for {symbol} over {period} period...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean data
            data = data.dropna()
            data.columns = [col.lower() for col in data.columns]
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_mean(self, prices: pd.Series) -> pd.Series:
        """
        Calculate long-term mean using EWMA and optional target mean
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        
        Returns:
        --------
        pd.Series
            Mean series
        """
        # Calculate EWMA
        ewma = prices.ewm(alpha=1-self.decay_factor, adjust=False).mean()
        
        if self.target_mean is not None:
            # Blend EWMA with target mean (70% EWMA, 30% target)
            # This allows the mean to evolve but stay anchored to fundamental value
            mean = 0.7 * ewma + 0.3 * self.target_mean
        else:
            # Use pure EWMA if no target mean specified
            mean = ewma
        
        return mean
    
    def calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling volatility using specified correlation length
        """
        return prices.rolling(window=self.corr_length).std()
    
    def calculate_speed_reversion(self, returns: pd.Series) -> pd.Series:
        """
        Calculate speed of mean reversion
        
        This method uses both statistical estimation and theoretical calibration
        """
        # Calculate statistical autocorrelation
        autocorr = returns.rolling(window=self.corr_length).apply(
            lambda x: x.autocorr(lag=1) if len(x) == self.corr_length else np.nan
        )
        
        # Calculate statistical theta
        autocorr_clipped = np.clip(autocorr, 0.0001, 0.9999)
        statistical_theta = -np.log(autocorr_clipped)
        
        # Convert annual theta to daily theta
        daily_theta_target = self.annual_theta / 252
        
        # Blend statistical and theoretical theta (60% statistical, 40% theoretical)
        # This provides stability while incorporating market dynamics
        calibrated_theta = 0.6 * statistical_theta + 0.4 * daily_theta_target
        
        return calibrated_theta
    
    def calculate_ou_bands(self, 
                          mean: pd.Series, 
                          volatility: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate upper and lower OU bands"""
        upper_band = mean + self.threshold * volatility
        lower_band = mean - self.threshold * volatility
        
        return upper_band, lower_band
    
    def calculate_deviation(self, 
                          prices: pd.Series, 
                          mean: pd.Series, 
                          volatility: pd.Series) -> pd.Series:
        """Calculate standardized deviation from mean"""
        return (prices - mean) / volatility
    
    def calculate_consecutive_bars(self, 
                                 deviation: pd.Series, 
                                 mean: pd.Series, 
                                 prices: pd.Series) -> pd.Series:
        """Calculate consecutive bars away from mean"""
        consecutive_bars = pd.Series(0, index=deviation.index)
        
        for i in range(1, len(deviation)):
            # Reset if price crosses mean
            if ((prices.iloc[i-1] < mean.iloc[i-1] and prices.iloc[i] >= mean.iloc[i]) or
                (prices.iloc[i-1] > mean.iloc[i-1] and prices.iloc[i] <= mean.iloc[i])):
                consecutive_bars.iloc[i] = 0
            elif abs(deviation.iloc[i]) > 0:
                consecutive_bars.iloc[i] = min(consecutive_bars.iloc[i-1] + 1, 400)
        
        return consecutive_bars
    
    
    def calculate_statistics(self, 
                           prices: pd.Series,
                           returns: pd.Series,
                           deviation: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive statistics for the analysis"""
        stats_dict = {}
        
        # Basic price statistics
        stats_dict['total_return'] = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        stats_dict['annualized_return'] = ((prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1) * 100
        stats_dict['volatility_annualized'] = returns.std() * np.sqrt(252) * 100
        stats_dict['sharpe_ratio'] = stats_dict['annualized_return'] / stats_dict['volatility_annualized'] if stats_dict['volatility_annualized'] > 0 else 0
        
        # OU process statistics
        stats_dict['mean_deviation'] = deviation.mean()
        stats_dict['std_deviation'] = deviation.std()
        stats_dict['max_deviation'] = deviation.max()
        stats_dict['min_deviation'] = deviation.min()
        
        
        # Mean reversion statistics
        stats_dict['mean_reversion_strength'] = abs(stats_dict['mean_deviation']) / stats_dict['std_deviation'] if stats_dict['std_deviation'] > 0 else 0
        
        # Parameters
        stats_dict['target_mean'] = self.target_mean
        stats_dict['annual_theta'] = self.annual_theta
        stats_dict['half_life_days'] = self.half_life_days
        
        return stats_dict
    
    def analyze_commodity(self, 
                        symbol: str, 
                        period: str = "5y",
                        interval: str = "1d") -> Dict:
        """
        Perform complete OU analysis on a commodity
        """
        logger.info(f"Starting OU analysis for {symbol}...")
        
        # Fetch data
        self.data = self.fetch_commodity_data(symbol, period, interval)
        prices = self.data['close']
        
        # Calculate components
        returns = self.calculate_returns(prices)
        mean = self.calculate_mean(prices)
        volatility = self.calculate_volatility(prices)
        speed_reversion = self.calculate_speed_reversion(returns)
        upper_band, lower_band = self.calculate_ou_bands(mean, volatility)
        deviation = self.calculate_deviation(prices, mean, volatility)
        consecutive_bars = self.calculate_consecutive_bars(deviation, mean, prices)
        statistics = self.calculate_statistics(prices, returns, deviation)
        
        # Store results
        self.ou_params = {
            'mean': mean,
            'volatility': volatility,
            'speed_reversion': speed_reversion
        }
        
        self.statistics = statistics
        
        # Prepare results
        results = {
            'symbol': symbol,
            'data': self.data,
            'prices': prices,
            'returns': returns,
            'mean': mean,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'deviation': deviation,
            'consecutive_bars': consecutive_bars,
            'statistics': statistics,
            'ou_params': self.ou_params,
            'parameters': {
                'target_mean': self.target_mean,
                'annual_theta': self.annual_theta,
                'half_life_days': self.half_life_days,
                'corr_length': self.corr_length,
                'threshold': self.threshold,
                'decay_factor': self.decay_factor
            }
        }
        
        logger.info(f"OU analysis completed for {symbol}")
        return results
    
    def create_visualization(self, 
                           results: Dict, 
                           save_path: str = None,
                           figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive visualization of OU analysis
        """
        symbol = results['symbol']
        prices = results['prices']
        mean = results['mean']
        upper_band = results['upper_band']
        lower_band = results['lower_band']
        deviation = results['deviation']
        consecutive_bars = results['consecutive_bars']
        statistics = results['statistics']
        parameters = results['parameters']
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1])
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot price and bands
        ax1.plot(prices.index, prices, 'k-', linewidth=1.5, label='Price', alpha=0.8)
        ax1.plot(mean.index, mean, 'b-', linewidth=2, label='Long-term Mean (μ)', alpha=0.9)
        ax1.plot(upper_band.index, upper_band, 'g--', linewidth=1.5, label=f'Upper Band (+{parameters["threshold"]}σ)', alpha=0.7)
        ax1.plot(lower_band.index, lower_band, 'r--', linewidth=1.5, label=f'Lower Band (-{parameters["threshold"]}σ)', alpha=0.7)
        
        # Add target mean line if specified
        if parameters['target_mean'] is not None:
            ax1.axhline(y=parameters['target_mean'], color='orange', 
                       linestyle=':', linewidth=2, alpha=0.8, 
                       label=f'Target Mean (${parameters["target_mean"]:.2f})')
        
        # Fill between bands with gradient effect
        ax1.fill_between(mean.index, mean, upper_band, 
                        where=(deviation > 0), 
                        color='green', alpha=0.1, interpolate=True)
        ax1.fill_between(mean.index, mean, lower_band, 
                        where=(deviation < 0), 
                        color='red', alpha=0.1, interpolate=True)
        
        
        ax1.set_title(f'OU Analysis: {symbol}', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Deviation chart
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        colors = ['red' if d < 0 else 'green' for d in deviation]
        ax2.bar(deviation.index, deviation, color=colors, alpha=0.6, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=parameters['threshold'], color='green', linestyle='--', alpha=0.7)
        ax2.axhline(y=-parameters['threshold'], color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Deviation (σ)', fontsize=12)
        ax2.set_title('Standardized Deviation from Mean', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Consecutive bars chart
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax3.plot(consecutive_bars.index, consecutive_bars, 'purple', linewidth=1.5)
        ax3.axhline(y=parameters['half_life_days'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Half-life ({parameters["half_life_days"]} days)')
        ax3.set_ylabel('Consecutive Bars', fontsize=12)
        ax3.set_title('Consecutive Bars Away from Mean', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Speed of reversion chart
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        ax4.plot(results['ou_params']['speed_reversion'].index, 
                results['ou_params']['speed_reversion'], 
                'orange', linewidth=1.5, label='Speed of Reversion (θ)')
        ax4.axhline(y=parameters['annual_theta']/252, color='red', linestyle='--', alpha=0.7,
                   label=f'Target Daily θ ({parameters["annual_theta"]/252:.4f})')
        ax4.set_ylabel('θ (daily)', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_title('Speed of Mean Reversion', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Statistics panel
        ax5 = fig.add_subplot(gs[:, 1])
        ax5.axis('off')
        
        # Create statistics text
        stats_text = f"""
        OU ANALYSIS STATISTICS
        =====================
        
        Symbol: {symbol}
        Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}
        Data Points: {len(prices):,}
        
        PARAMETERS
        ----------
        Target Mean: {"${:.2f}".format(parameters['target_mean']) if parameters['target_mean'] is not None else "Data-driven"}
        Annual θ: {parameters['annual_theta']:.1f}
        Half-life: {parameters['half_life_days']} days
        Correlation Length: {parameters['corr_length']} days
        Threshold: {parameters['threshold']}σ
        Decay Factor: {parameters['decay_factor']:.3f}
        
        PRICE STATISTICS
        ----------------
        Total Return: {statistics['total_return']:.2f}%
        Annualized Return: {statistics['annualized_return']:.2f}%
        Annualized Volatility: {statistics['volatility_annualized']:.2f}%
        Sharpe Ratio: {statistics['sharpe_ratio']:.3f}
        
        MEAN REVERSION ANALYSIS
        -----------------------
        Mean Deviation: {statistics['mean_deviation']:.3f}
        Std Deviation: {statistics['std_deviation']:.3f}
        Max Deviation: {statistics['max_deviation']:.3f}
        Min Deviation: {statistics['min_deviation']:.3f}
        Mean Reversion Strength: {statistics['mean_reversion_strength']:.3f}
        
        
        INTERPRETATION
        --------------
        """
        
        # Add interpretation
        if statistics['mean_reversion_strength'] > 1.0:
            stats_text += "Strong mean reversion detected\n"
        elif statistics['mean_reversion_strength'] > 0.5:
            stats_text += "Moderate mean reversion\n"
        else:
            stats_text += "Weak mean reversion\n"
        
        
        # Add half-life interpretation
        if parameters['half_life_days'] <= 100:
            stats_text += "Fast mean reversion\n"
        else:
            stats_text += "Slower mean reversion\n"
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def run_analysis(self, 
                    symbols: List[str], 
                    period: str = "5y",
                    save_plots: bool = True) -> Dict[str, Dict]:
        """
        Run OU analysis on multiple commodities
        """
        all_results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Analyzing {symbol}...")
                results = self.analyze_commodity(symbol, period)
                all_results[symbol] = results
                
                if save_plots:
                    filename = f"ou_analysis_{symbol.replace('=', '_').replace('^', '_')}.png"
                    self.create_visualization(results, save_path=filename)
                
                # Print summary
                stats = results['statistics']
                params = results['parameters']
                print(f"\n{'='*70}")
                print(f"OU ANALYSIS: {symbol}")
                print(f"{'='*70}")
                print(f"Target Mean: ${params['target_mean']:.2f}" if params['target_mean'] else "Target Mean: Data-driven")
                print(f"Annual θ: {params['annual_theta']:.1f} (Half-life: {params['half_life_days']} days)")
                print(f"Total Return: {stats['total_return']:.2f}%")
                print(f"Annualized Return: {stats['annualized_return']:.2f}%")
                print(f"Volatility: {stats['volatility_annualized']:.2f}%")
                print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
                print(f"Mean Reversion Strength: {stats['mean_reversion_strength']:.3f}")
                print(f"{'='*70}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        return all_results


def main():
    """
    Main function to run the OU analysis
    """
    print("Generic Ornstein-Uhlenbeck Process Analyzer")
    print("=" * 60)
    print("Customizable Parameters for Commodity Analysis")
    print("=" * 60)
    
    # Initialize analyzer with default parameters
    analyzer = GenericOUAnalyzer(
        target_mean=None,      # Data-driven mean
        annual_theta=2.5,      # 2.5 annual mean reversion rate
        corr_length=100,       # 100-day correlation window
        threshold=2.0,         # 2 standard deviations
        decay_factor=0.95      # EWMA decay factor
    )
    
    # Define commodity symbols for analysis
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
    
    print(f"\nAnalyzing {len(commodities)} commodities...")
    print("Commodities:", list(commodities.keys()))
    
    # Run analysis
    results = analyzer.run_analysis(
        symbols=list(commodities.values()),
        period="5y",
        save_plots=True
    )
    
    # Create summary comparison
    print(f"\n{'='*90}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Commodity':<15} {'Target μ':<10} {'Annual θ':<10} {'Half-life':<10} {'Return%':<10} {'Vol%':<8} {'Sharpe':<8} {'OU Strength':<12}")
    print("-" * 90)
    
    for name, symbol in commodities.items():
        if symbol in results:
            stats = results[symbol]['statistics']
            params = results[symbol]['parameters']
            target_mean = f"${params['target_mean']:.1f}" if params['target_mean'] else "Data-driven"
            print(f"{name:<15} {target_mean:<10} {params['annual_theta']:<10.1f} {params['half_life_days']:<10} {stats['total_return']:<10.2f} {stats['volatility_annualized']:<8.2f} "
                  f"{stats['sharpe_ratio']:<8.3f} {stats['mean_reversion_strength']:<12.3f}")
    
    print(f"\n{'='*90}")
    print("Analysis completed successfully!")
    print("Individual commodity charts saved as PNG files.")
    print("Use the GenericOUAnalyzer class to customize parameters for specific commodities.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()