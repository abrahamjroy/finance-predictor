import warnings
import os

# Suppress statsmodels warnings about date frequency
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*date index has been provided.*')
warnings.filterwarnings('ignore', message='.*No supported index is available.*')

# Suppress XGBoost device mismatch warnings (one-time warning already shown)
os.environ['XGBOOST_VERBOSITY'] = '1'  # 0=silent, 1=warning, 2=info, 3=debug
