import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import mixed_precision

# Enable mixed precision
#mixed_precision.set_global_policy('mixed_float16')

# Define RSLU activation function (Rectified Stretch Linear Unit)
def rslu(x, alpha=0.5, max_value=None):
    """
    Rectified Stretch Linear Unit activation function.
    
    Args:
        x: Input tensor
        alpha: Stretch parameter (default: 0.5)
        max_value: Maximum output value (default: None)
        
    Returns:
        Tensor with RSLU activation applied
    """
    x_pos = tf.nn.relu(x)
    x_neg = -alpha * tf.nn.relu(-x)
    result = x_pos + x_neg
    
    if max_value is not None:
        result = tf.minimum(result, max_value)
        
    return result

# Register RSLU as a custom object
tf.keras.utils.get_custom_objects()['rslu'] = rslu

class TradingPlatform:
    def __init__(self, symbol, start_date, end_date, sequence_length=120):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.sp500_data = None  # Add S&P500 data
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length
        
    def fetch_data(self):
        """Fetch historical data including S&P500"""
        # Fetch stock data
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=self.start_date, end=self.end_date)
        
        # Fetch S&P500 data
        sp500 = yf.Ticker('^GSPC')
        self.sp500_data = sp500.history(start=self.start_date, end=self.end_date)
        
        return self.data
    
    def prepare_features(self, window=14):
        """Enhanced feature engineering to improve prediction accuracy and reduce MAPE"""
        df = self.data.copy()
        sp500 = self.sp500_data.copy()
        
        # Keep core price data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Store original close prices before scaling
        self.original_close_prices = df['Close'].copy()
        
        #----- ENHANCED BASIC FEATURES -----
        
        # Log-transform prices to make them more stationary (better for ML)
        for col in ['Open', 'High', 'Low', 'Close']:
            df[f'{col}_Log'] = np.log1p(df[col])
        
        # Calculate returns - better predictor than raw prices
        df['Returns'] = df['Close'].pct_change()
        
        # Log returns (more statistically well-behaved)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Add percentage change features instead of raw prices
        df['Open_pct'] = df['Open'] / df['Close'].shift(1) - 1
        df['High_pct'] = df['High'] / df['Close'].shift(1) - 1
        df['Low_pct'] = df['Low'] / df['Close'].shift(1) - 1
        df['Close_pct'] = df['Close'] / df['Close'].shift(1) - 1
        
        # Log transform volume - better for scaling
        df['Volume'] = np.log1p(df['Volume'])
        
        #----- ENHANCED TECHNICAL INDICATORS -----
        
        # RSI with multiple timeframes
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            # Simple Moving Average
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            # Exponential Moving Average
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            # Express as percentage difference from current price
            df[f'Price_to_SMA_{period}'] = (df['Close'] / df[f'SMA_{period}'] - 1) * 100
            df[f'Price_to_EMA_{period}'] = (df['Close'] / df[f'EMA_{period}'] - 1) * 100
        
        # Calculate EMA crossovers separately after all EMAs are created
        # This ensures we don't try to access EMAs that don't exist yet
        for period in [5, 10, 20]:
            slow_period = period * 2
            # Make sure both EMAs exist before creating crossover feature
            if f'EMA_{period}' in df.columns and f'EMA_{slow_period}' in df.columns:
                df[f'EMA_Cross_{period}_{slow_period}'] = df[f'EMA_{period}'] - df[f'EMA_{slow_period}']
        
        # MACD with multiple parameters
        for (fast, slow, signal) in [(12, 26, 9), (8, 21, 5), (5, 35, 5)]:
            exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
            df[f'MACD_{fast}_{slow}'] = exp1 - exp2
            df[f'Signal_{fast}_{slow}_{signal}'] = df[f'MACD_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
            df[f'MACD_Hist_{fast}_{slow}_{signal}'] = df[f'MACD_{fast}_{slow}'] - df[f'Signal_{fast}_{slow}_{signal}']
        
        # Keep standard MACD terms for compatibility
        df['MACD'] = df['MACD_12_26']
        df['Signal_Line'] = df['Signal_12_26_9']
        df['MACD_Hist'] = df['MACD_Hist_12_26_9']
        
        # Volatility indicators
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
        df['ATR'] = df['Daily_Range'].rolling(window=14).mean()
        # Normalized ATR
        df['ATR_pct'] = df['ATR'] / df['Close'] * 100
        
        # Bollinger Bands with multiple timeframes
        for period in [10, 20, 50]:
            for stdev in [1.5, 2.0, 2.5]:
                df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'BB_Std_{period}'] = df['Close'].rolling(window=period).std()
                df[f'BB_Upper_{period}_{stdev}'] = df[f'BB_Middle_{period}'] + stdev * df[f'BB_Std_{period}']
                df[f'BB_Lower_{period}_{stdev}'] = df[f'BB_Middle_{period}'] - stdev * df[f'BB_Std_{period}']
                df[f'BB_Width_{period}_{stdev}'] = (df[f'BB_Upper_{period}_{stdev}'] - df[f'BB_Lower_{period}_{stdev}']) / df[f'BB_Middle_{period}']
                df[f'BB_Position_{period}_{stdev}'] = (df['Close'] - df[f'BB_Lower_{period}_{stdev}']) / (df[f'BB_Upper_{period}_{stdev}'] - df[f'BB_Lower_{period}_{stdev}'])
        
        # Use standard terms for compatibility
        df['BB_Middle'] = df['BB_Middle_20']
        df['BB_Std'] = df['BB_Std_20']
        df['BB_Upper'] = df['BB_Upper_20_2.0']
        df['BB_Lower'] = df['BB_Lower_20_2.0']
        df['BB_Width'] = df['BB_Width_20_2.0']
        df['BB_Position'] = df['BB_Position_20_2.0']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Rate'] = df['Volume'] / df['Volume_MA']
        
        # Volume trends
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Change_5d'] = df['Volume'].pct_change(periods=5)
        df['Volume_Change_10d'] = df['Volume'].pct_change(periods=10)
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_Rate'] = df['OBV'].pct_change(5)
        
        #----- ENHANCED MARKET RELATION FEATURES -----
        
        # Add S&P500 correlation and relative strength
        df['SP500_Corr'] = df['Close'].rolling(window=60).corr(sp500['Close'])
        for period in [5, 10, 20, 60]:
            df[f'SP500_RelStrength_{period}'] = (df['Close'] / df['Close'].shift(period)) / (sp500['Close'] / sp500['Close'].shift(period))
        df['SP500_RelStrength'] = df['SP500_RelStrength_20']  # For compatibility
        
        # Price momentum features for multiple timeframes
        for period in [1, 3, 5, 10, 20, 60]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Acceleration of momentum
        for period in [5, 10, 20]:
            df[f'Acceleration_{period}'] = df[f'Momentum_{period}'] - df[f'Momentum_{period}'].shift(period)
        
        # Simple lag features for core indicators
        for col in ['Close_pct', 'Returns', 'RSI_14', 'MACD']:
            for i in [1, 2, 3, 5, 10]:
                df[f'{col}_Lag_{i}'] = df[col].shift(i)
        
        # Advanced indicator: Rate of Change for multiple periods
        for period in [1, 3, 5, 10, 20, 60]:
            df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
        
        #----- ENHANCED TREND INDICATORS -----
        
        # Directional movement indicators with multiple timeframes
        for period in [7, 14, 28]:
            # Calculate true range
            df['TR'] = np.maximum(df['High'] - df['Low'], 
                        np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                  abs(df['Low'] - df['Close'].shift(1))))
            
            # Calculate directional movement
            df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                                  np.maximum(df['High'] - df['High'].shift(1), 0), 0)
            df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                                   np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
            
            # Smooth using EMA
            df[f'TR_{period}'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
            df[f'DMplus_{period}'] = df['DMplus'].ewm(alpha=1/period, adjust=False).mean()
            df[f'DMminus_{period}'] = df['DMminus'].ewm(alpha=1/period, adjust=False).mean()
            
            # Calculate directional indicators
            df[f'DIplus_{period}'] = 100 * df[f'DMplus_{period}'] / df[f'TR_{period}']
            df[f'DIminus_{period}'] = 100 * df[f'DMminus_{period}'] / df[f'TR_{period}']
            
            # Calculate directional index and average directional index
            df[f'DX_{period}'] = 100 * abs(df[f'DIplus_{period}'] - df[f'DIminus_{period}']) / (df[f'DIplus_{period}'] + df[f'DIminus_{period}'])
            df[f'ADX_{period}'] = df[f'DX_{period}'].ewm(alpha=1/period, adjust=False).mean()
        
        # Keltner Channels
        for period in [10, 20]:
            df[f'KC_Middle_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'KC_Width_{period}'] = df[f'ATR'] * 2
            df[f'KC_Upper_{period}'] = df[f'KC_Middle_{period}'] + df[f'KC_Width_{period}']
            df[f'KC_Lower_{period}'] = df[f'KC_Middle_{period}'] - df[f'KC_Width_{period}']
            
            # Squeeze momentum indicator (Bollinger Bands vs. Keltner Channels)
            bb = 2 * df['Close'].rolling(window=period).std()
            df[f'Squeeze_{period}'] = (df[f'BB_Upper_{period}_2.0'] - df[f'BB_Lower_{period}_2.0']) / (df[f'KC_Upper_{period}'] - df[f'KC_Lower_{period}'])
        
        #----- PATTERN RECOGNITION -----
        
        # Detect doji pattern (indecision)
        df['Doji'] = (abs(df['Open'] - df['Close']) / (df['High'] - df['Low'])) < 0.1
        
        # Detect gap up and gap down
        df['Gap_Up'] = df['Low'] > df['High'].shift(1)
        df['Gap_Down'] = df['High'] < df['Low'].shift(1)
        
        # Price reversals - detect turning points
        df['Reversal_Up'] = (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2)) & (df['Close'].shift(2) < df['Close'].shift(3))
        df['Reversal_Down'] = (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2)) & (df['Close'].shift(2) > df['Close'].shift(3))
        
        # Prepare target variables
        df['Target_pct'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = df['Close'].shift(-1)
        
        # Smooth target for training (helps reduce noise and improve generalization)
        df['Target_pct_smooth'] = df['Target_pct'].rolling(window=3, center=True).mean()
        
        # Store the target before scaling for later use
        self.target_series = df['Target'].copy()
        self.target_pct_series = df['Target_pct'].copy()
        
        # Drop auxiliary columns used for calculations
        cols_to_drop = ['TR', 'DMplus', 'DMminus']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Store a copy of the dataframe before scaling for reference
        self.df_before_scaling = df.copy()
        
        # Use a separate scaler for price-related columns to better handle their range
        price_columns = [col for col in df.columns if any(term in col for term in ['Open', 'High', 'Low', 'Close', 'SMA', 'EMA', 'BB_', 'KC_'])]
        price_columns = [col for col in price_columns if 'pct' not in col and 'Log' not in col and not col.startswith('Price_to_')]
        price_columns += ['Target']
        
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        df[price_columns] = self.price_scaler.fit_transform(df[price_columns])
        
        # Use a different scaler for percentage changes
        pct_columns = [col for col in df.columns if any(x in col for x in ['pct', 'Returns', 'Log_Returns', 'Momentum_', 'ROC_', 'Price_to_'])]
        self.pct_scaler = MinMaxScaler(feature_range=(-1, 1))
        df[pct_columns] = self.pct_scaler.fit_transform(df[pct_columns])
        
        # Scale other features
        used_columns = price_columns + pct_columns
        non_price_columns = [col for col in df.columns if col not in used_columns]
        
        # Handle boolean columns
        bool_columns = ['Doji', 'Gap_Up', 'Gap_Down', 'Reversal_Up', 'Reversal_Down']
        for col in bool_columns:
            if col in non_price_columns:
                df[col] = df[col].astype(float)
                non_price_columns.remove(col)
        
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        df[non_price_columns] = self.feature_scaler.fit_transform(df[non_price_columns])
        
        return df
    
    def _calculate_rsi(self, prices, window):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_sequences(self, df):
        """Enhanced sequence creation with feature selection for better MAPE"""
        # Expanded set of important predictive features
        features = [
            # Price change features (more stable than raw prices)
            'Open_pct', 'High_pct', 'Low_pct', 'Close_pct',
            'Returns', 'Log_Returns',
            
            # Main technical indicators - multiple timeframes
            'RSI_7', 'RSI_14', 'RSI_21',
            'MACD', 'Signal_Line', 'MACD_Hist',
            'MACD_8_21', 'Signal_8_21_5',  # Only include if sure these exist
            
            # Multiple timeframe moving averages
            'Price_to_SMA_20', 'Price_to_SMA_50', 'Price_to_SMA_200',
            'Price_to_EMA_20', 'Price_to_EMA_50',
            'EMA_Cross_5_10', 'EMA_Cross_10_20',  # Only include verified crossovers
            
            # Volatility indicators
            'Daily_Range', 'ATR', 'ATR_pct',
            
            # Bollinger Bands
            'BB_Width', 'BB_Position',
            'BB_Width_20_2.0', 'BB_Position_20_2.0',
            
            # Volume analysis
            'Volume', 'Volume_MA', 'Volume_Rate', 'Volume_Change',
            
            # S&P500 correlation and relative strength
            'SP500_Corr', 'SP500_RelStrength',
            
            # Momentum and Rate of Change
            'Momentum_5', 'Momentum_10', 'Momentum_20', 'Momentum_60',
            'ROC_5', 'ROC_10', 'ROC_20',
            
            # Directional movement indicators
            'DIplus_14', 'DIminus_14', 'ADX_14',
            'DIplus_28', 'DIminus_28', 'ADX_28',
            
            # Pattern recognition
            'Doji', 'Gap_Up', 'Gap_Down',
            
            # Lag features for core indicators
            'Close_pct_Lag_1', 'Close_pct_Lag_2', 'Close_pct_Lag_3',
            'Returns_Lag_1', 'Returns_Lag_2',
            'RSI_14_Lag_1', 'RSI_14_Lag_2',
            'MACD_Lag_1', 'MACD_Lag_2'
        ]
        
        # Filter to include only features that exist in the dataframe
        filtered_features = [feature for feature in features if feature in df.columns]
        
        # Handle case where some features don't exist
        missing_features = set(features) - set(filtered_features)
        if missing_features:
            print(f"Warning: {len(missing_features)} requested features are not in the dataframe: {', '.join(missing_features)}")
            print(f"Proceeding with {len(filtered_features)} available features.")
        
        print("\nCore features used for training:")
        for i, feature in enumerate(filtered_features, 1):
            print(f"{i}. {feature}")
        
        # Use standard float32 for all data
        X = df[filtered_features].values.astype('float32')
        
        # Use percentage change for target (can use smoothed version if available)
        if 'Target_pct_smooth' in df.columns:
            y = df['Target_pct_smooth'].values.astype('float32')
            print("Using smoothed target percentage change")
        else:
            y = df['Target_pct'].values.astype('float32')
            print("Using raw target percentage change")
        
        X_seq, y_seq = [], []
        for i in range(len(df) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        
        # Convert to numpy arrays
        X_seq = np.array(X_seq, dtype='float32')
        y_seq = np.array(y_seq, dtype='float32')
        
        print(f"\nInput shape: {X_seq.shape}")
        print(f"Output shape: {y_seq.shape}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Number of features: {len(filtered_features)}")
        print(f"Input dtype: {X_seq.dtype}")
        print(f"Output dtype: {y_seq.dtype}")
            
        return X_seq, y_seq, filtered_features
    
    def build_lstm_model(self, input_shape):
        """Enhanced LSTM model architecture with feature interaction layers and ensemble components"""
        # Use lighter regularization to prevent underfitting while controlling overfitting
        kernel_regularizer = L1L2(l1=1e-10, l2=1e-8)
        recurrent_regularizer = L1L2(l1=1e-10, l2=1e-8)
        
        # Input layer
        inputs = tf.keras.Input(shape=input_shape)
        
        # Feature extraction block with multiple kernels for capturing different patterns
        conv1 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
        
        # Concatenate different kernel outputs
        conv_concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        conv_bn = BatchNormalization()(conv_concat)
        
        # First LSTM path (captures short-term patterns)
        lstm1 = LSTM(128, return_sequences=True, 
                   kernel_regularizer=kernel_regularizer,
                   recurrent_regularizer=recurrent_regularizer,
                   recurrent_dropout=0.0,  # Avoid recurrent dropout for stability
                   kernel_initializer='glorot_uniform')(conv_bn)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.2)(lstm1)
        
        # Second LSTM path (captures medium-term patterns)
        lstm2 = LSTM(96, return_sequences=True,
                   kernel_regularizer=kernel_regularizer,
                   recurrent_regularizer=recurrent_regularizer,
                   kernel_initializer='glorot_uniform')(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Attention mechanism - highlights important time steps
        attention = tf.keras.layers.Dense(1, activation='tanh')(lstm2)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(96)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention to LSTM output
        lstm_attention = tf.keras.layers.Multiply()([lstm2, attention])
        
        # Global feature extraction
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm_attention)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm_attention)
        
        # Concatenate different pooling strategies
        pooling = tf.keras.layers.Concatenate()([max_pool, avg_pool])
        pooling_bn = BatchNormalization()(pooling)
        
        # Feature interaction layer with deeper MLP (enhances non-linear feature combinations)
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(pooling_bn)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Ensemble outputs approach - multiple prediction heads
        output1 = Dense(32, activation='relu')(x)
        output1 = Dense(1)(output1)
        
        output2 = Dense(32, activation='relu')(x)
        output2 = Dense(1)(output2)
        
        output3 = Dense(32, activation='relu')(x)
        output3 = Dense(1)(output3)
        
        # Combine the ensemble predictions with learnable weights
        ensemble_outputs = tf.keras.layers.Concatenate()([output1, output2, output3])
        weighted_sum = Dense(1, 
                            activation='linear',
                            kernel_initializer=tf.constant_initializer([0.33, 0.33, 0.34]),
                            use_bias=False)(ensemble_outputs)
        
        # Build model
        model = tf.keras.Model(inputs=inputs, outputs=weighted_sum)
        
        # Enhanced optimizer with gradient clipping and warm-up
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Higher initial learning rate with stepped decay
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Using Huber loss which is more robust to outliers than MSE
        # This helps with financial data that often has spikes/outliers
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),  # Huber loss is more robust to outliers
            metrics=['mae', 'mse']
        )
        
        model.summary()
        return model
    
    def train_model(self, df):
        """Enhanced training process with advanced techniques to reduce MAPE"""
        # Create sequences with enhanced features
        X_seq, y_seq, features = self.create_sequences(df)
        self.features = features  # Store features for prediction
        
        # Implement time-based split to maintain temporal order
        train_size = int(len(X_seq) * 0.7)  # 70% for training
        val_size = int(len(X_seq) * 0.15)   # 15% for validation
        X_train = X_seq[:train_size]
        X_val = X_seq[train_size:train_size+val_size]
        X_test = X_seq[train_size+val_size:]
        y_train = y_seq[:train_size]
        y_val = y_seq[train_size:train_size+val_size]
        y_test = y_seq[train_size+val_size:]
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Create the model
        self.model = self.build_lstm_model(input_shape=(self.sequence_length, X_train.shape[2]))
        
        # Data augmentation: add small noise to provide robustness
        # This improves generalization by creating slightly different versions of the same data
        def create_augmented_data(X, y, noise_level=0.05, num_augmented=1):
            """Create augmented versions of the data with small amounts of noise"""
            X_aug, y_aug = [], []
            
            for _ in range(num_augmented):
                # Add small random noise
                noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
                X_noisy = X + noise
                
                X_aug.append(X_noisy)
                y_aug.append(y)
            
            # Combine original and augmented data
            X_combined = np.vstack([X] + X_aug)
            y_combined = np.concatenate([y] + y_aug)
            
            return X_combined, y_combined
        
        # Only augment training data, not validation or test
        # Only apply to more recent half of the training data for relevance
        recent_half = len(X_train) // 2
        X_aug, y_aug = create_augmented_data(
            X_train[recent_half:], 
            y_train[recent_half:],
            noise_level=0.03,
            num_augmented=1
        )
        
        X_train_full = np.vstack([X_train[:recent_half], X_aug])
        y_train_full = np.concatenate([y_train[:recent_half], y_aug])
        
        # Apply sample weights to give more importance to recent data
        sample_weights = np.ones(len(X_train_full))
        # Original data gets gradually increasing weights
        orig_weights = np.linspace(0.7, 1.0, recent_half)
        # Augmented data gets uniform weights based on recency
        aug_weights = np.linspace(0.9, 1.0, len(X_aug))
        
        sample_weights[:recent_half] = orig_weights
        sample_weights[recent_half:] = aug_weights
        
        # One-cycle learning rate schedule
        def one_cycle_lr(epoch, lr):
            total_epochs = 200  # Changed from 1 to a more appropriate value
            # Warm up phase (first 5% of epochs)
            if epoch < total_epochs * 0.05:
                return 0.0001 + (0.001 - 0.0001) * (epoch / (total_epochs * 0.05))
            # First half of training - increase LR
            elif epoch < total_epochs * 0.4:
                return 0.001
            # Second half - gradually decrease LR
            elif epoch < total_epochs * 0.7:
                return 0.0005
            elif epoch < total_epochs * 0.9:
                return 0.0001
            else:
                return 0.00005
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(one_cycle_lr)
        
        # Early stopping with longer patience for financial data
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,  # Longer patience for financial time series
            restore_best_weights=True,
            min_delta=1e-6,
            verbose=1
        )
        
        # Model checkpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Reduce learning rate on plateau as additional control
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train with full training data
        print("Training model with enhanced techniques...")
        history = self.model.fit(
            X_train_full, y_train_full,
            epochs=200,  # Changed from 1 to 200 epochs
            batch_size=16,  # Smaller batch size for better generalization
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint, lr_scheduler, reduce_lr],
            verbose=1,
            shuffle=True,
            sample_weight=sample_weights
        )
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        
        print("\nTraining Summary:")
        print(f"Total Epochs Trained: {len(history.history['loss'])}")
        print(f"Best Training Loss: {min(history.history['loss']):.6f}")
        print(f"Best Validation Loss: {min(history.history['val_loss']):.6f}")
        print(f"Final Training Loss: {train_loss[0]:.6f}")
        print(f"Final Validation Loss: {val_loss[0]:.6f}")
        print(f"Final Testing Loss: {test_loss[0]:.6f}")
        
        # Calculate and display error metrics in terms of real prices
        self._evaluate_price_predictions(X_test, y_test)
        
        return history
    
    def _evaluate_price_predictions(self, X_test, y_test):
        """Enhanced evaluation method with focus on reducing MAPE"""
        # Make predictions (percentage change)
        pct_change_pred = self.model.predict(X_test, verbose=0)
        
        # Get the corresponding original prices from before target date
        idx_offset = len(self.df_before_scaling) - len(y_test)
        actual_prices = []
        predicted_prices = []
        dates = []
        
        # Convert percentage predictions to actual prices
        for i in range(len(y_test)):
            idx = idx_offset + i
            
            # Ensure the index is within bounds of the dataframe
            if idx < len(self.df_before_scaling):
                base_price = self.df_before_scaling['Close'].iloc[idx]
                actual_pct = y_test[i]
                pred_pct = pct_change_pred[i][0]
                
                # Actual next day price
                actual_price = base_price * (1 + actual_pct)
                # Predicted next day price
                pred_price = base_price * (1 + pred_pct)
                
                actual_prices.append(actual_price)
                predicted_prices.append(pred_price)
                
                # Add date only if the next index exists
                if idx + 1 < len(self.df_before_scaling):
                    dates.append(self.df_before_scaling.index[idx + 1])  # Use date of predicted day
        
        # Convert to numpy arrays
        actual_prices = np.array(actual_prices)
        predicted_prices = np.array(predicted_prices)
        
        # Calculate error metrics
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        
        # Calculate MAPE with protection against division by zero and NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            abs_percentage_errors = np.abs((actual_prices - predicted_prices) / actual_prices)
            # Replace infinities and NaNs with a high value (100%)
            abs_percentage_errors = np.nan_to_num(abs_percentage_errors, nan=1.0, posinf=1.0, neginf=1.0)
            # Cap extremely high values
            abs_percentage_errors = np.minimum(abs_percentage_errors, 1.0)
            mape = 100.0 * np.mean(abs_percentage_errors)
        
        rmse = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
        
        # Median absolute percentage error (often more representative than mean)
        med_abs_pct_error = 100.0 * np.median(abs_percentage_errors)
        
        # Calculate directional accuracy (how often we get the direction right)
        actual_directions = np.sign(actual_prices - np.array([self.df_before_scaling['Close'].iloc[idx_offset + i] for i in range(len(y_test))]))
        pred_directions = np.sign(predicted_prices - np.array([self.df_before_scaling['Close'].iloc[idx_offset + i] for i in range(len(y_test))]))
        directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        
        print("\nReal Price Prediction Metrics:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"Median Absolute Percentage Error: {med_abs_pct_error:.2f}%")
        print(f"Root Mean Squared Error: ${rmse:.2f}")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Store the error metrics for later use
        self.error_metrics = {
            'mae': mae,
            'mape': mape,
            'medape': med_abs_pct_error,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
        
        # Plot actual vs predicted prices for a subset of the test data
        plt.figure(figsize=(14, 10))
        
        # Recent data plot - line chart
        plt.subplot(2, 1, 1)
        subset_size = min(50, len(actual_prices))
        plt.plot(actual_prices[-subset_size:], label='Actual Prices', marker='o', markersize=5, linewidth=2)
        plt.plot(predicted_prices[-subset_size:], label='Predicted Prices', marker='x', markersize=5, linewidth=2)
        plt.title('Actual vs Predicted Prices (Recent Test Data)')
        plt.xlabel('Sample Index')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot of actual vs predicted to visualize correlation
        plt.subplot(2, 1, 2)
        plt.scatter(actual_prices, predicted_prices, alpha=0.5)
        plt.plot([min(actual_prices), max(actual_prices)], [min(actual_prices), max(actual_prices)], 'r--')  # Perfect prediction line
        plt.title('Actual vs Predicted Scatter Plot')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # If we have date information, plot time series
        if len(dates) > 0:
            plt.figure(figsize=(14, 7))
            plt.plot(dates[-subset_size:], actual_prices[-subset_size:], label='Actual', marker='o', markersize=4)
            plt.plot(dates[-subset_size:], predicted_prices[-subset_size:], label='Predicted', marker='x', markersize=4)
            plt.title(f'{self.symbol} Actual vs Predicted Prices Over Time')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        return self.error_metrics
    
    def predict_next_day(self, df):
        """Enhanced prediction method with ensemble approach to reduce MAPE"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get the features used in training
        features = self.features
        
        # Make sure we're using the same features as in training
        print(f"Number of features for prediction: {len(features)}")
        
        # Multi-sequence ensemble approach for more robust predictions
        predictions = []
        base_sequences = []
        
        # Create multiple sequences with slight variations for ensemble prediction
        # This adds robustness by considering different views of the input data
        
        # 1. Standard sequence
        last_sequence = df[features].values[-self.sequence_length:]
        base_sequences.append(last_sequence)
        
        # 2. Sequence with more weight on recent data
        weighted_sequence = last_sequence.copy()
        # Apply increasing weights to more recent data
        recency_weights = np.linspace(0.8, 1.2, self.sequence_length)
        for i in range(self.sequence_length):
            weighted_sequence[i] = weighted_sequence[i] * recency_weights[i]
        base_sequences.append(weighted_sequence)
        
        # 3. Sequence with small random noise for robustness
        noise_sequence = last_sequence.copy()
        noise = np.random.normal(0, 0.02, noise_sequence.shape).astype(np.float32)
        noise_sequence = noise_sequence + noise
        base_sequences.append(noise_sequence)
        
        # Make predictions with each sequence
        for sequence in base_sequences:
            # Reshape for LSTM input
            input_seq = sequence.reshape((1, self.sequence_length, len(features)))
            
            # Make prediction (percentage change)
            pred = self.model.predict(input_seq, verbose=0)[0][0]
            predictions.append(pred)
        
        # Ensemble the predictions with weighted average
        # Give more weight to the standard prediction (0.6) and less to variations (0.2 each)
        ensemble_weights = [0.6, 0.2, 0.2]
        predicted_pct_change = np.average(predictions, weights=ensemble_weights)
        
        # Get the last actual closing price for reference
        last_price = self.original_close_prices.iloc[-1]
        print(f"Last closing price: ${last_price:.2f}")
        
        # Calculate the predicted next day price using percentage change
        next_day_prediction = last_price * (1 + predicted_pct_change)
        
        # Calculate percentage change
        pct_change = predicted_pct_change * 100
        print(f"Predicted change: {pct_change:.2f}%")
        
        # Individual prediction results
        print("\nEnsemble prediction details:")
        for i, pred in enumerate(predictions):
            pred_price = last_price * (1 + pred)
            weight = ensemble_weights[i]
            print(f"Model {i+1}: ${pred_price:.2f} (weight: {weight:.2f})")
        
        # Calculate confidence interval based on model's historical error
        if hasattr(self, 'error_metrics'):
            # Prefer median error when available as it's less affected by outliers
            if 'medape' in self.error_metrics:
                mape = self.error_metrics['medape']
            else:
                mape = self.error_metrics['mape']
            
            # Ensure the MAPE is valid and not too large
            if np.isfinite(mape) and mape > 0 and mape < 50:
                # Use MAPE for error margin calculation
                error_margin = (mape / 100) * last_price
            else:
                # Fallback to a reasonable default if MAPE is invalid
                error_margin = 0.02 * last_price
                print("Note: Using default error margin due to invalid MAPE value.")
            
            confidence = 0.9  # 90% confidence
            
            # Check if we have directional accuracy
            if 'directional_accuracy' in self.error_metrics:
                dir_acc = self.error_metrics['directional_accuracy']
                print(f"Direction prediction confidence: {dir_acc:.2f}%")
                
                # Interpret the direction
                if predicted_pct_change > 0:
                    direction = "UP ↑"
                else:
                    direction = "DOWN ↓"
                    
                print(f"Predicted direction: {direction}")
            
            print(f"\nPrediction with {confidence*100:.0f}% confidence interval:")
            print(f"Next day closing price: ${next_day_prediction:.2f} ± ${error_margin:.2f}")
            print(f"Range: ${next_day_prediction - error_margin:.2f} to ${next_day_prediction + error_margin:.2f}")
        else:
            # Fallback to default error margin if no error metrics available
            error_margin = 0.02 * last_price
            print(f"\nPrediction with default confidence interval:")
            print(f"Next day closing price: ${next_day_prediction:.2f} ± ${error_margin:.2f}")
            print(f"Range: ${next_day_prediction - error_margin:.2f} to ${next_day_prediction + error_margin:.2f}")
        
        return next_day_prediction
    
    def plot_training_history(self, history):
        """Enhanced training history visualization with more metrics"""
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # MAE plot
        plt.subplot(2, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        if 'lr' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['lr'], label='Learning Rate')
            plt.title('Learning Rate over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
        
        # Loss distribution plot
        plt.subplot(2, 2, 4)
        plt.hist(history.history['loss'], bins=50, alpha=0.5, label='Training Loss')
        plt.hist(history.history['val_loss'], bins=50, alpha=0.5, label='Validation Loss')
        plt.title('Loss Distribution')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_results(self, df):
        """Plot actual prices with technical indicators"""
        # Use the original dataframe before scaling for plotting
        orig_df = self.df_before_scaling
        
        # First plot: Price and SMA
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Price and indicators plot with original values
        # Fix the dimension mismatch by aligning indices
        # Get the original_close_prices but only for the indices that exist in the processed dataframe
        aligned_prices = self.original_close_prices.loc[self.original_close_prices.index.isin(orig_df.index)]
        
        ax1.plot(orig_df.index, orig_df['Close'], label='Close Price', color='blue')
        ax1.plot(orig_df.index, orig_df['SMA_200'], label='200-day SMA', color='orange', alpha=0.7)
        
        # Add Bollinger Bands
        ax1.plot(orig_df.index, orig_df['BB_Upper'], label='Bollinger Upper', color='red', linestyle='--', alpha=0.3)
        ax1.plot(orig_df.index, orig_df['BB_Lower'], label='Bollinger Lower', color='green', linestyle='--', alpha=0.3)
        ax1.fill_between(orig_df.index, orig_df['BB_Upper'], orig_df['BB_Lower'], color='gray', alpha=0.1)
        
        # Add MACD
        ax2.plot(orig_df.index, orig_df['MACD'], label='MACD', color='blue', alpha=0.7)
        ax2.plot(orig_df.index, orig_df['Signal_Line'], label='Signal Line', color='red', alpha=0.7)
        ax2.bar(orig_df.index, orig_df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('MACD')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.legend(loc='upper left')
        
        # Focus on the most recent year for better visualization
        if len(orig_df) > 252:  # Approximately one trading year
            start_idx = len(orig_df) - 252
            ax1.set_xlim(orig_df.index[start_idx], orig_df.index[-1])
            ax2.set_xlim(orig_df.index[start_idx], orig_df.index[-1])
        
        ax1.set_title(f'{self.symbol} Stock Price and Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Second plot: RSI and Volume
        fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 12))
        
        # RSI plot using original unscaled data
        ax3.plot(orig_df.index, orig_df['RSI_14'], label='RSI', color='purple')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.fill_between(orig_df.index, 70, 30, color='gray', alpha=0.1)
        ax3.set_title('Relative Strength Index (RSI)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('RSI Value')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Volume plot - need to convert from log back to original
        volume_orig = np.expm1(orig_df['Volume'])
        volume_ma_orig = np.expm1(orig_df['Volume_MA'])
        
        ax4.bar(orig_df.index, volume_orig, label='Volume', color='green', alpha=0.5)
        ax4.plot(orig_df.index, volume_ma_orig, label='Volume MA', color='red', alpha=0.7)
        ax4.set_title('Trading Volume')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Volume')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Focus on the most recent year for better visualization
        if len(orig_df) > 252:  # Approximately one trading year
            start_idx = len(orig_df) - 252
            ax3.set_xlim(orig_df.index[start_idx], orig_df.index[-1])
            ax4.set_xlim(orig_df.index[start_idx], orig_df.index[-1])
        
        plt.tight_layout()
        plt.show()
        
        # Third plot: Momentum and ADX indicators
        fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Momentum indicators
        ax5.plot(orig_df.index, orig_df['Momentum_5'], label='5-day Momentum', color='blue')
        ax5.plot(orig_df.index, orig_df['Momentum_20'], label='20-day Momentum', color='orange')
        ax5.plot(orig_df.index, orig_df['Momentum_60'], label='60-day Momentum', color='green')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_title('Price Momentum')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Momentum (%)')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # ADX and Directional Indicators
        ax6.plot(orig_df.index, orig_df['ADX_14'], label='ADX (14)', color='black', linewidth=2)
        ax6.plot(orig_df.index, orig_df['DIplus_14'], label='DI+ (14)', color='green')
        ax6.plot(orig_df.index, orig_df['DIminus_14'], label='DI- (14)', color='red')
        ax6.axhline(y=25, color='gray', linestyle='--', alpha=0.5)  # Strong trend threshold
        ax6.set_title('Average Directional Index (ADX)')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Value')
        ax6.set_ylim(0, 60)
        ax6.legend(loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        # Focus on the most recent year for better visualization
        if len(orig_df) > 252:  # Approximately one trading year
            start_idx = len(orig_df) - 252
            ax5.set_xlim(orig_df.index[start_idx], orig_df.index[-1])
            ax6.set_xlim(orig_df.index[start_idx], orig_df.index[-1])
        
        plt.tight_layout()
        plt.show()



def main():
    # Initialize platform with longer history for LSTM
    symbol = 'GOOGL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2000)  # Get more historical data for LSTM
    
    platform = TradingPlatform(symbol, start_date, end_date, sequence_length=120)
    
    # Fetch and prepare data
    data = platform.fetch_data()
    prepared_data = platform.prepare_features()
    
    # Train model
    history = platform.train_model(prepared_data)
    
    # Make prediction for next day
    next_day_prediction = platform.predict_next_day(prepared_data)
    print(f"\nPredicted next day closing price: ${next_day_prediction:.2f}")
    
    # Plot results
    platform.plot_results(prepared_data)

if __name__ == "__main__":
    main()