"""
USA Options AI - ML Prediction Dashboard
Shows real-time predictions from trained neural network models with options strategies
"""
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ML libraries
try:
    from tensorflow import keras
    import yfinance as yf
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML libraries not available. Using demo mode.")

# Initialize Dash app
app = dash.Dash(__name__, title="USA Options AI - Predictions")
server = app.server

# Configuration - Dynamically load available symbols
DATA_DIR = 'data'
MODELS_DIR = 'saved_models'

def get_available_symbols():
    """Get list of symbols with trained models"""
    symbols = []
    if os.path.exists(MODELS_DIR):
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.h5')]
        symbols = [f.replace('_model.h5', '') for f in model_files]
    return sorted(symbols) if symbols else ['MSFT', 'GOOGL', 'AMZN', 'TSLA']

SYMBOLS = get_available_symbols()
print(f"Loaded {len(SYMBOLS)} trained models")

def load_data_for_symbol(symbol):
    """Load historical data from CSV file"""
    try:
        file_path = os.path.join(DATA_DIR, f'{symbol}_data.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            return df
        else:
            print(f"Data file not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        return None

def load_model_for_symbol(symbol):
    """Load trained model for symbol"""
    if not ML_AVAILABLE:
        return None
    try:
        model_path = os.path.join(MODELS_DIR, f'{symbol}_model.h5')
        if os.path.exists(model_path):
            # Load model without compiling (skip metric deserialization)
            model = keras.models.load_model(model_path, compile=False)
            # Recompile with correct metric
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        else:
            print(f"Model not found: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model for {symbol}: {e}")
        return None

def prepare_features(df):
    """Prepare features for prediction (same as training)"""
    feature_cols = ['open', 'high', 'low', 'volume', 'sma_5', 'sma_10', 
                   'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                   'bb_upper', 'bb_lower', 'volatility']
    
    available_cols = [col for col in feature_cols if col in df.columns]
    if len(available_cols) == 0:
        return np.array([]), np.array([False] * len(df))
    
    X = df[available_cols].values
    
    # Remove any rows with NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    
    return X, mask

def make_predictions(symbol, df, model):
    """Generate predictions using trained model"""
    try:
        X, mask = prepare_features(df)
        if len(X) == 0:
            return None, None
        
        # Normalize features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled, verbose=0)
        
        # Create prediction series aligned with data
        pred_series = pd.Series(index=df.index[mask], data=predictions.flatten())
        
        return pred_series, None
    except Exception as e:
        return None, str(e)

def get_latest_price(symbol):
    """Get latest price from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return None

def get_stock_info(symbol):
    """Get stock name and current price"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get('longName', symbol)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        return name, current_price
    except:
        return symbol, None

def calculate_options_strategy(symbol, current_price, predicted_prices):
    """Calculate options trading strategy with multiple targets"""
    if current_price is None or predicted_prices is None or len(predicted_prices) == 0:
        return []
    
    # Get next 3 price predictions
    recent_predictions = predicted_prices[-3:].values if len(predicted_prices) >= 3 else predicted_prices.values
    
    # Calculate targets based on predicted movement
    avg_predicted = np.mean(recent_predictions)
    price_change = avg_predicted - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Determine option type (CALL if upward, PUT if downward)
    option_type = "CALL" if price_change > 0 else "PUT"
    
    # Calculate strike prices (ATM, OTM)
    if option_type == "CALL":
        strike_price = round(current_price * 1.02, 2)  # 2% OTM
    else:
        strike_price = round(current_price * 0.98, 2)  # 2% OTM
    
    # Generate expiry dates (weekly options)
    today = datetime.now()
    expiry_dates = [
        (today + timedelta(days=7)).strftime('%Y-%m-%d'),   # 1 week
        (today + timedelta(days=14)).strftime('%Y-%m-%d'),  # 2 weeks
        (today + timedelta(days=30)).strftime('%Y-%m-%d')   # 1 month
    ]
    
    # Calculate 3 targets based on volatility
    volatility = np.std(predicted_prices[-10:].values) if len(predicted_prices) >= 10 else np.std(predicted_prices.values)
    target1 = round(current_price + (price_change * 0.5), 2)  # Conservative (50% of predicted move)
    target2 = round(current_price + price_change, 2)  # Moderate (full predicted move)
    target3 = round(current_price + (price_change * 1.5), 2)  # Aggressive (150% of predicted move)
    
    # Calculate confidence levels based on prediction consistency
    recent_accuracy = []
    for i in range(min(10, len(predicted_prices)-1)):
        idx = -(i+2)
        pred = predicted_prices.iloc[idx]
        actual = predicted_prices.iloc[idx+1] if idx+1 < 0 else predicted_prices.iloc[-1]
        if actual != 0:
            error = abs(pred - actual) / actual
            recent_accuracy.append(1 - min(error, 1))
    
    base_confidence = np.mean(recent_accuracy) * 100 if recent_accuracy else 75
    confidence1 = min(round(base_confidence, 1), 95)  # Conservative target - highest confidence
    confidence2 = min(round(base_confidence * 0.85, 1), 85)  # Moderate target
    confidence3 = min(round(base_confidence * 0.65, 1), 70)  # Aggressive target
    
    # Calculate capital required (estimated option premium)
    # Simple Black-Scholes approximation for option premium
    days_to_expiry = [7, 14, 30]
    option_premiums = []
    
    for days in days_to_expiry:
        # Rough option premium estimate: intrinsic value + time value
        intrinsic = max(0, abs(strike_price - current_price))
        time_value = volatility * np.sqrt(days/365) * current_price * 0.1
        premium = intrinsic + time_value
        option_premiums.append(round(premium, 2))
    
    # Capital required for 1 contract (100 shares)
    capital_required = [round(premium * 100, 2) for premium in option_premiums]
    
    # Risk level calculation based on volatility and price movement
    if abs(price_change_pct) < 2:
        risk_level = "LOW"
    elif abs(price_change_pct) < 5:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    # Risk/Reward ratio calculation
    potential_profit = [abs(target - strike_price) * 100 for target in [target1, target2, target3]]
    risk_reward_ratios = [round(profit / cap, 2) if cap > 0 else 0 
                         for profit, cap in zip(potential_profit, capital_required)]
    
    # Get stock name once (with error handling)
    try:
        stock_name = get_stock_info(symbol)[0]
    except:
        stock_name = symbol
    
    # Create options strategies for each expiry
    strategies = []
    for i, expiry in enumerate(expiry_dates):
        strategy = {
            'Symbol': symbol,
            'Name': stock_name,
            'Option Type': option_type,
            'Expiry Date': expiry,
            'Stock Price': f"${current_price:.2f}",
            'Strike Price': f"${strike_price:.2f}",
            'Target 1': f"${target1:.2f}",
            'Target 2': f"${target2:.2f}",
            'Target 3': f"${target3:.2f}",
            'Confidence 1': f"{confidence1:.1f}%",
            'Confidence 2': f"{confidence2:.1f}%",
            'Confidence 3': f"{confidence3:.1f}%",
            'Capital Required': f"${capital_required[i]:.2f}",
            'Risk Level': risk_level,
            'Risk/Reward': f"{risk_reward_ratios[i]:.2f}",
            'Predicted Move': f"{price_change_pct:+.2f}%"
        }
        strategies.append(strategy)
    
    return strategies

# Define layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🤖 USA Options AI - Options Trading Strategies", 
                    style={'margin': 0, 'color': '#ffffff'}),
            html.P("ML-Powered Options Predictions with Risk Analysis & Multiple Targets", 
                   style={'margin': '5px 0 0 0', 'color': '#ecf0f1', 'fontSize': '16px'})
        ], style={'flex': '1'}),
        html.Div([
            html.Div(id='last-update', 
                     style={'color': '#ecf0f1', 'fontSize': '14px', 'textAlign': 'right'})
        ])
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '30px',
        'borderRadius': '15px',
        'marginBottom': '20px',
        'display': 'flex',
        'alignItems': 'center',
        'boxShadow': '0 10px 30px rgba(0,0,0,0.2)'
    }),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("📊 Select Stock Symbol:", 
                      style={'fontWeight': 'bold', 'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[{'label': f'{symbol}', 'value': symbol} for symbol in SYMBOLS],
                value=None,  # Start with ALL stocks view
                style={'borderRadius': '8px'},
                searchable=True,
                placeholder="🏠 All Stocks (Select one for details)"
            ),
        ], style={'width': '300px'}),
        
        html.Div([
            html.Button('🔄 Refresh Predictions', 
                       id='refresh-button', 
                       n_clicks=0,
                       style={
                           'backgroundColor': '#667eea',
                           'color': 'white',
                           'border': 'none',
                           'padding': '12px 30px',
                           'borderRadius': '8px',
                           'cursor': 'pointer',
                           'fontSize': '16px',
                           'fontWeight': 'bold',
                           'marginLeft': '20px',
                           'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                       })
        ])
    ], style={
        'display': 'flex',
        'alignItems': 'flex-end',
        'padding': '20px',
        'backgroundColor': '#ffffff',
        'borderRadius': '10px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
    }),
    
    # Status Cards
    html.Div(id='status-cards', style={'marginBottom': '20px'}),
    
    # Main Chart - Price + Predictions
    html.Div([
        html.H3("📈 Price History & ML Predictions", 
                style={'marginBottom': '15px', 'color': '#2c3e50'}),
        dcc.Graph(id='prediction-chart', 
                 config={'displayModeBar': True, 'displaylogo': False},
                 style={'height': '500px'})
    ], style={
        'backgroundColor': '#ffffff',
        'padding': '25px',
        'borderRadius': '10px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
    }),
    
    # Prediction Accuracy Metrics
    html.Div([
        html.Div([
            html.H3("🎯 Prediction Performance", 
                    style={'marginBottom': '15px', 'color': '#2c3e50'}),
            html.Div(id='accuracy-metrics')
        ], style={'width': '48%'}),
        
        html.Div([
            html.H3("📊 Technical Indicators", 
                    style={'marginBottom': '15px', 'color': '#2c3e50'}),
            dcc.Graph(id='indicators-chart', 
                     config={'displayModeBar': False},
                     style={'height': '300px'})
        ], style={'width': '48%'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'marginBottom': '20px'
    }),
    
    # Options Trading Strategies Table
    html.Div([
        html.H3("📊 Options Trading Strategies", 
                style={'marginBottom': '15px', 'color': '#2c3e50', 'fontSize': '26px', 'fontWeight': 'bold'}),
        html.P("ML-Generated Options Strategies with Multiple Targets & Risk Analysis", 
               style={'color': '#7f8c8d', 'marginBottom': '20px', 'fontSize': '16px'}),
        html.Div(id='predictions-table')
    ], style={
        'backgroundColor': '#ffffff',
        'padding': '25px',
        'borderRadius': '10px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    }),
    
    # Summary Statistics Table
    html.Div([
        html.H3("📊 Prediction Summary Statistics", 
                style={'marginBottom': '15px', 'color': '#2c3e50'}),
        html.Div(id='summary-stats')
    ], style={
        'backgroundColor': '#ffffff',
        'padding': '25px',
        'borderRadius': '10px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
    }),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every 60 seconds
        n_intervals=0
    )
], style={
    'fontFamily': 'Arial, sans-serif',
    'padding': '20px',
    'backgroundColor': '#f5f6fa'
})

def generate_all_stocks_view():
    """Generate view showing options strategies for ALL stocks"""
    print("Generating ALL STOCKS view...")
    
    all_strategies = []
    total_stocks = len(SYMBOLS)
    processed = 0
    
    # Process each stock - Get ALL expiry dates (3 per stock)
    for sym in SYMBOLS:
        try:
            df = load_data_for_symbol(sym)
            model = load_model_for_symbol(sym)
            
            if df is not None and model is not None:
                current_price = df['close'].iloc[-1]
                predictions, _ = make_predictions(sym, df, model)
                
                if predictions is not None:
                    strategies = calculate_options_strategy(sym, current_price, predictions)
                    if strategies:
                        # Add ALL strategies (all 3 expiry dates)
                        all_strategies.extend(strategies)
                        processed += 1
        except Exception as e:
            print(f"Error processing {sym}: {e}")
            continue
    
    print(f"Processed {processed}/{total_stocks} stocks with {len(all_strategies)} total strategies")
    
    # Empty/minimal components for chart and status cards
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        template='plotly_white',
        height=1,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    
    # Empty status cards
    status_cards = html.Div(style={'display': 'none'})
    
    # Create options table for ALL stocks
    if all_strategies:
        options_df = pd.DataFrame(all_strategies)
        
        predictions_table = html.Div([
            html.H4("📋 ALL STOCKS - Options Trading Strategies (Nearest Expiry)", 
                   style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
            html.P(f"Showing {len(all_strategies)} options strategies across all stocks. Select a stock above for detailed multi-expiry analysis.",
                   style={'color': '#7f8c8d', 'textAlign': 'center', 'marginBottom': '20px'}),
            dash_table.DataTable(
                data=options_df.to_dict('records'),
                columns=[
                    {'name': 'Symbol', 'id': 'Symbol'},
                    {'name': 'Name', 'id': 'Name'},
                    {'name': 'Option Type', 'id': 'Option Type'},
                    {'name': 'Expiry Date', 'id': 'Expiry Date'},
                    {'name': 'Stock Price', 'id': 'Stock Price'},
                    {'name': 'Strike Price', 'id': 'Strike Price'},
                    {'name': 'Target 1', 'id': 'Target 1'},
                    {'name': 'Target 2', 'id': 'Target 2'},
                    {'name': 'Target 3', 'id': 'Target 3'},
                    {'name': 'Confidence 1', 'id': 'Confidence 1'},
                    {'name': 'Confidence 2', 'id': 'Confidence 2'},
                    {'name': 'Confidence 3', 'id': 'Confidence 3'},
                    {'name': 'Capital Required', 'id': 'Capital Required'},
                    {'name': 'Risk Level', 'id': 'Risk Level'},
                    {'name': 'Risk/Reward', 'id': 'Risk/Reward'},
                    {'name': 'Predicted Move', 'id': 'Predicted Move'}
                ],
                style_cell={
                    'textAlign': 'center',
                    'padding': '12px',
                    'fontSize': '13px',
                    'fontFamily': 'Arial, sans-serif',
                    'minWidth': '100px'
                },
                style_header={
                    'backgroundColor': '#667eea',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'fontSize': '14px',
                    'padding': '15px',
                    'border': '2px solid #667eea'
                },
                style_data_conditional=[
                    # CALL options - green
                    {
                        'if': {'column_id': 'Option Type', 'filter_query': '{Option Type} = "CALL"'},
                        'backgroundColor': '#d4edda',
                        'color': '#155724',
                        'fontWeight': 'bold'
                    },
                    # PUT options - red
                    {
                        'if': {'column_id': 'Option Type', 'filter_query': '{Option Type} = "PUT"'},
                        'backgroundColor': '#f8d7da',
                        'color': '#721c24',
                        'fontWeight': 'bold'
                    },
                    # Risk levels
                    {
                        'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "LOW"'},
                        'backgroundColor': '#d4edda',
                        'color': '#155724',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "MEDIUM"'},
                        'backgroundColor': '#fff3cd',
                        'color': '#856404',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "HIGH"'},
                        'backgroundColor': '#f8d7da',
                        'color': '#721c24',
                        'fontWeight': 'bold'
                    },
                    # Confidence highlighting
                    {
                        'if': {'column_id': ['Confidence 1', 'Confidence 2', 'Confidence 3']},
                        'fontWeight': 'bold',
                        'color': '#9b59b6'
                    },
                    # Target columns
                    {
                        'if': {'column_id': ['Target 1', 'Target 2', 'Target 3']},
                        'backgroundColor': '#e8f5e9',
                        'fontWeight': 'bold'
                    },
                    # Row alternation
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ],
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'maxHeight': '600px',
                    'border': '2px solid #667eea',
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.2)'
                },
                style_data={
                    'border': '1px solid #dee2e6',
                    'lineHeight': '1.5'
                },
                page_action='none',
                fixed_rows={'headers': True},
                fixed_columns={'headers': True, 'data': 1},
                sort_action='native',
                filter_action='native'
            )
        ])
    else:
        predictions_table = html.Div("No strategies generated", style={'padding': '20px', 'color': '#e74c3c'})
    
    # Return 7 outputs - all minimal except the predictions table
    empty_indicators_fig = go.Figure()
    empty_indicators_fig.update_layout(
        title="",
        template='plotly_white',
        height=1,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    
    return (
        empty_fig,  # prediction-chart (hidden)
        status_cards,  # status-cards (hidden)
        html.Div(style={'display': 'none'}),  # accuracy-metrics (hidden)
        empty_indicators_fig,  # indicators-chart (hidden)
        predictions_table,  # predictions-table (MAIN CONTENT)
        html.Div(style={'display': 'none'}),  # summary-stats (hidden)
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  # last-update
    )

@app.callback(
    [Output('prediction-chart', 'figure'),
     Output('status-cards', 'children'),
     Output('accuracy-metrics', 'children'),
     Output('indicators-chart', 'figure'),
     Output('predictions-table', 'children'),
     Output('summary-stats', 'children'),
     Output('last-update', 'children')],
    [Input('symbol-dropdown', 'value'),
     Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(symbol, n_clicks, n_intervals):
    """Update all dashboard components"""
    
    # If no symbol selected, show ALL stocks options strategies
    if symbol is None:
        return generate_all_stocks_view()
    
    # Load data
    df = load_data_for_symbol(symbol)
    model = load_model_for_symbol(symbol)
    
    if df is None:
        # Return empty state
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            xaxis_title="Date",
            yaxis_title="Value"
        )
        return empty_fig, html.Div("No data"), html.Div("No metrics"), empty_fig, html.Div("No predictions"), html.Div("No stats"), ""
    
    # Get predictions if model available
    predictions = None
    if model and ML_AVAILABLE:
        predictions, error = make_predictions(symbol, df, model)
        if error:
            print(f"Prediction error: {error}")
    
    # Create main chart
    fig_main = go.Figure()
    
    # Add actual prices
    fig_main.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['close'].iloc[-100:],
        mode='lines',
        name='Actual Price',
        line=dict(color='#3498db', width=2)
    ))
    
    # Add predictions if available
    if predictions is not None:
        fig_main.add_trace(go.Scatter(
            x=predictions.index[-100:],
            y=predictions.iloc[-100:],
            mode='lines',
            name='ML Prediction',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
    
    # Add moving averages
    if 'sma_20' in df.columns:
        fig_main.add_trace(go.Scatter(
            x=df.index[-100:],
            y=df['sma_20'].iloc[-100:],
            mode='lines',
            name='SMA 20',
            line=dict(color='#2ecc71', width=1, dash='dot'),
            opacity=0.6
        ))
    
    fig_main.update_layout(
        title=f"{symbol} - Price History & Predictions",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    # Status cards
    current_price = df['close'].iloc[-1]
    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
    price_change_pct = (price_change / df['close'].iloc[-2]) * 100
    
    latest_prediction = predictions.iloc[-1] if predictions is not None else None
    prediction_diff = (latest_prediction - current_price) if latest_prediction else 0
    
    latest_live_price = get_latest_price(symbol)
    
    status_cards = html.Div([
        # Current Price
        html.Div([
            html.Div("💵 Current Price", style={'fontSize': '14px', 'color': '#7f8c8d', 'marginBottom': '5px'}),
            html.Div(f"${current_price:.2f}", style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#2c3e50'}),
            html.Div(
                f"{'▲' if price_change >= 0 else '▼'} ${abs(price_change):.2f} ({price_change_pct:+.2f}%)",
                style={'fontSize': '14px', 'color': '#27ae60' if price_change >= 0 else '#e74c3c', 'marginTop': '5px'}
            )
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '10px',
            'flex': '1',
            'marginRight': '15px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)',
            'border': '2px solid #3498db'
        }),
        
        # ML Prediction
        html.Div([
            html.Div("🤖 ML Prediction", style={'fontSize': '14px', 'color': '#7f8c8d', 'marginBottom': '5px'}),
            html.Div(
                f"${latest_prediction:.2f}" if latest_prediction else "N/A",
                style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#2c3e50'}
            ),
            html.Div(
                f"{'▲' if prediction_diff >= 0 else '▼'} ${abs(prediction_diff):.2f} vs current" if latest_prediction else "Model loading...",
                style={'fontSize': '14px', 'color': '#9b59b6', 'marginTop': '5px'}
            )
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '10px',
            'flex': '1',
            'marginRight': '15px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)',
            'border': '2px solid #9b59b6'
        }),
        
        # Live Price
        html.Div([
            html.Div("📡 Live Price", style={'fontSize': '14px', 'color': '#7f8c8d', 'marginBottom': '5px'}),
            html.Div(
                f"${latest_live_price:.2f}" if latest_live_price else "Fetching...",
                style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#2c3e50'}
            ),
            html.Div(
                "Real-time Yahoo Finance",
                style={'fontSize': '14px', 'color': '#95a5a6', 'marginTop': '5px'}
            )
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '10px',
            'flex': '1',
            'marginRight': '15px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)',
            'border': '2px solid #27ae60'
        }),
        
        # Model Status
        html.Div([
            html.Div("⚙️ Model Status", style={'fontSize': '14px', 'color': '#7f8c8d', 'marginBottom': '5px'}),
            html.Div(
                "ACTIVE" if model else "NOT LOADED",
                style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#27ae60' if model else '#e74c3c'}
            ),
            html.Div(
                f"{len(df)} data points" if df is not None else "No data",
                style={'fontSize': '14px', 'color': '#95a5a6', 'marginTop': '5px'}
            )
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '10px',
            'flex': '1',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)',
            'border': '2px solid #f39c12'
        })
    ], style={'display': 'flex'})
    
    # Accuracy metrics
    if predictions is not None:
        # Calculate errors on last 50 points
        recent_actual = df['close'].iloc[-50:]
        recent_pred = predictions.iloc[-50:]
        
        # Align indices
        common_idx = recent_actual.index.intersection(recent_pred.index)
        if len(common_idx) > 0:
            actual_aligned = recent_actual.loc[common_idx]
            pred_aligned = recent_pred.loc[common_idx]
            
            mae = np.mean(np.abs(actual_aligned - pred_aligned))
            mape = np.mean(np.abs((actual_aligned - pred_aligned) / actual_aligned)) * 100
            rmse = np.sqrt(np.mean((actual_aligned - pred_aligned) ** 2))
            
            accuracy_metrics = html.Div([
                html.Div([
                    html.Div("Mean Absolute Error (MAE)", style={'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.Div(f"${mae:.2f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#3498db'})
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Div("Mean Absolute Percentage Error (MAPE)", style={'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.Div(f"{mape:.2f}%", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#e67e22'})
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Div("Root Mean Square Error (RMSE)", style={'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.Div(f"${rmse:.2f}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#9b59b6'})
                ])
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
            })
        else:
            accuracy_metrics = html.Div("Insufficient data for accuracy calculation")
    else:
        accuracy_metrics = html.Div("Model predictions not available", 
                                    style={'color': '#e74c3c', 'padding': '20px'})
    
    # Indicators chart (RSI)
    fig_indicators = go.Figure()
    if 'rsi' in df.columns:
        fig_indicators.add_trace(go.Scatter(
            x=df.index[-100:],
            y=df['rsi'].iloc[-100:],
            mode='lines',
            name='RSI',
            line=dict(color='#9b59b6', width=2),
            fill='tozeroy'
        ))
        fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        fig_indicators.update_layout(
            title="Relative Strength Index (RSI)",
            yaxis_title="RSI",
            template='plotly_white',
            showlegend=False,
            height=300
        )
    
    # Predictions table (Options Trading Strategy)
    if predictions is not None and current_price:
        # Generate options strategies
        strategies = calculate_options_strategy(symbol, current_price, predictions)
        
        if strategies:
            # Create DataFrame from strategies
            options_df = pd.DataFrame(strategies)
            
            predictions_table = html.Div([
                html.H4("📋 Options Trading Strategies - Multiple Expiry Dates", 
                       style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
                dash_table.DataTable(
                    data=options_df.to_dict('records'),
                    columns=[
                        {'name': 'Symbol', 'id': 'Symbol'},
                        {'name': 'Name', 'id': 'Name'},
                        {'name': 'Option Type', 'id': 'Option Type'},
                        {'name': 'Expiry Date', 'id': 'Expiry Date'},
                        {'name': 'Stock Price', 'id': 'Stock Price'},
                        {'name': 'Strike Price', 'id': 'Strike Price'},
                        {'name': 'Target 1', 'id': 'Target 1'},
                        {'name': 'Target 2', 'id': 'Target 2'},
                        {'name': 'Target 3', 'id': 'Target 3'},
                        {'name': 'Confidence 1', 'id': 'Confidence 1'},
                        {'name': 'Confidence 2', 'id': 'Confidence 2'},
                        {'name': 'Confidence 3', 'id': 'Confidence 3'},
                        {'name': 'Capital Required', 'id': 'Capital Required'},
                        {'name': 'Risk Level', 'id': 'Risk Level'},
                        {'name': 'Risk/Reward', 'id': 'Risk/Reward'},
                        {'name': 'Predicted Move', 'id': 'Predicted Move'}
                    ],
                    style_cell={
                        'textAlign': 'center',
                        'padding': '15px',
                        'fontSize': '13px',
                        'fontFamily': 'Arial, sans-serif',
                        'minWidth': '120px',
                        'maxWidth': '180px',
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_header={
                        'backgroundColor': '#667eea',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize': '14px',
                        'padding': '18px',
                        'textAlign': 'center',
                        'border': '1px solid #5568d3'
                    },
                    style_data_conditional=[
                        # Option Type styling
                        {
                            'if': {'column_id': 'Option Type', 'filter_query': '{Option Type} = "CALL"'},
                            'backgroundColor': '#d4edda',
                            'color': '#155724',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'column_id': 'Option Type', 'filter_query': '{Option Type} = "PUT"'},
                            'backgroundColor': '#f8d7da',
                            'color': '#721c24',
                            'fontWeight': 'bold'
                        },
                        # Risk Level styling
                        {
                            'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "LOW"'},
                            'backgroundColor': '#d4edda',
                            'color': '#155724',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "MEDIUM"'},
                            'backgroundColor': '#fff3cd',
                            'color': '#856404',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "HIGH"'},
                            'backgroundColor': '#f8d7da',
                            'color': '#721c24',
                            'fontWeight': 'bold'
                        },
                        # Confidence highlighting
                        {
                            'if': {'column_id': ['Confidence 1', 'Confidence 2', 'Confidence 3']},
                            'fontWeight': 'bold',
                            'color': '#9b59b6'
                        },
                        # Row alternation
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        },
                        # Target columns
                        {
                            'if': {'column_id': ['Target 1', 'Target 2', 'Target 3']},
                            'backgroundColor': '#e8f5e9',
                            'fontWeight': 'bold'
                        },
                        # Risk/Reward highlighting
                        {
                            'if': {'column_id': 'Risk/Reward'},
                            'fontWeight': 'bold',
                            'fontSize': '14px'
                        }
                    ],
                    style_table={
                        'overflowX': 'auto',
                        'border': '2px solid #667eea',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.2)'
                    },
                    style_data={
                        'border': '1px solid #dee2e6',
                        'lineHeight': '1.5'
                    },
                    page_action='none',
                    fixed_rows={'headers': True},
                    fixed_columns={'headers': True, 'data': 2}
                ),
                
                # Legend and Notes
                html.Div([
                    html.H5("📖 Strategy Guide:", style={'color': '#2c3e50', 'marginTop': '25px', 'marginBottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.Span("🎯 Target 1: ", style={'fontWeight': 'bold', 'color': '#27ae60'}),
                            html.Span("Conservative target (50% of predicted move)")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("🎯 Target 2: ", style={'fontWeight': 'bold', 'color': '#f39c12'}),
                            html.Span("Moderate target (100% of predicted move)")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("🎯 Target 3: ", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                            html.Span("Aggressive target (150% of predicted move)")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("💰 Capital Required: ", style={'fontWeight': 'bold', 'color': '#3498db'}),
                            html.Span("Estimated option premium × 100 shares (1 contract)")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("📊 Risk/Reward: ", style={'fontWeight': 'bold', 'color': '#9b59b6'}),
                            html.Span("Potential profit ÷ capital required (higher is better)")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Span("⚠️ Disclaimer: ", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                            html.Span("These are ML-based predictions for educational purposes. Always do your own research and never invest more than you can afford to lose.", 
                                    style={'fontStyle': 'italic'})
                        ], style={'marginTop': '15px', 'padding': '15px', 'backgroundColor': '#fff3cd', 
                                'borderRadius': '8px', 'border': '1px solid #ffc107'})
                    ], style={'fontSize': '14px', 'lineHeight': '1.8'})
                ])
            ])
            
            # Summary Statistics for Options
            summary_stats = html.Div([
                html.Div([
                    html.Div([
                        html.H4("📊 Strategy Summary", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                        dash_table.DataTable(
                            data=[{
                                'Metric': 'Recommended Strategy',
                                'Value': strategies[0]['Option Type']
                            }, {
                                'Metric': 'Predicted Price Move',
                                'Value': strategies[0]['Predicted Move']
                            }, {
                                'Metric': 'Best Expiry (Lowest Capital)',
                                'Value': min(strategies, key=lambda x: float(x['Capital Required'].replace('$','').replace(',','')))['Expiry Date']
                            }, {
                                'Metric': 'Best Risk/Reward',
                                'Value': f"{max([float(s['Risk/Reward']) for s in strategies]):.2f}"
                            }, {
                                'Metric': 'Avg Confidence Level',
                                'Value': f"{np.mean([float(s['Confidence 1'].replace('%','')) for s in strategies]):.1f}%"
                            }],
                            columns=[
                                {'name': 'Metric', 'id': 'Metric'},
                                {'name': 'Value', 'id': 'Value'}
                            ],
                            style_cell={
                                'textAlign': 'left',
                                'padding': '12px',
                                'fontSize': '14px'
                            },
                            style_header={
                                'backgroundColor': '#667eea',
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f8f9fa'
                                }
                            ]
                        )
                    ], style={'width': '100%'})
                ])
            ])
        else:
            predictions_table = html.Div("Unable to generate options strategies", 
                                        style={'padding': '20px', 'color': '#e74c3c', 'textAlign': 'center', 'fontSize': '16px'})
            summary_stats = html.Div("No statistics available")
    else:
        predictions_table = html.Div("No predictions available - Model may not be loaded or price data unavailable", 
                                     style={'padding': '20px', 'color': '#7f8c8d', 'textAlign': 'center', 'fontSize': '16px'})
        summary_stats = html.Div("No statistics available", 
                                style={'padding': '20px', 'color': '#7f8c8d', 'textAlign': 'center'})
    
    # Last update time
    last_update = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return fig_main, status_cards, accuracy_metrics, fig_indicators, predictions_table, summary_stats, last_update

if __name__ == '__main__':
    print("\n" + "="*70)
    print("USA OPTIONS AI - ML PREDICTION DASHBOARD")
    print("="*70)
    print(f"Total Models: {len(SYMBOLS)}")
    if len(SYMBOLS) <= 10:
        print(f"Symbols: {', '.join(SYMBOLS)}")
    else:
        print(f"First 10: {', '.join(SYMBOLS[:10])}")
        print(f"   ... and {len(SYMBOLS) - 10} more")
    print(f"ML Status: {'ENABLED' if ML_AVAILABLE else 'DEMO MODE'}")
    print(f"Dashboard URL: http://localhost:8050")
    print("="*70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=8050)
