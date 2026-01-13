"""
Interactive Plotly/Dash Visualization Dashboard
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from storage.timescaledb_client import TimescaleDBManager
from storage.influxdb_client import InfluxDBManager


# Initialize Dash app
app = dash.Dash(__name__, title="USA Options AI Dashboard")
server = app.server

# Initialize database clients
timescale_db = TimescaleDBManager()
influx_db = InfluxDBManager()

# Define layout
app.layout = html.Div([
    html.H1("USA Options AI - Interactive Analytics Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Select Symbol:"),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[
                    {'label': 'AAPL', 'value': 'AAPL'},
                    {'label': 'MSFT', 'value': 'MSFT'},
                    {'label': 'GOOGL', 'value': 'GOOGL'},
                    {'label': 'AMZN', 'value': 'AMZN'},
                    {'label': 'TSLA', 'value': 'TSLA'}
                ],
                value='AAPL',
                style={'width': '200px'}
            )
        ], style={'display': 'inline-block', 'marginRight': 20}),
        
        html.Div([
            html.Label("Time Range:"),
            dcc.Dropdown(
                id='timerange-dropdown',
                options=[
                    {'label': '1 Hour', 'value': 1},
                    {'label': '1 Day', 'value': 24},
                    {'label': '1 Week', 'value': 168},
                    {'label': '1 Month', 'value': 720}
                ],
                value=24,
                style={'width': '150px'}
            )
        ], style={'display': 'inline-block', 'marginRight': 20}),
        
        html.Button('Refresh', id='refresh-button', n_clicks=0,
                   style={'marginTop': 20})
    ], style={'marginBottom': 30, 'padding': 20, 'backgroundColor': '#ecf0f1'}),
    
    # Main content area
    html.Div([
        # Row 1: Price chart and Greeks
        html.Div([
            html.Div([
                dcc.Graph(id='price-chart')
            ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                dcc.Graph(id='greeks-gauge')
            ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ]),
        
        # Row 2: Options chain and volatility
        html.Div([
            html.Div([
                dcc.Graph(id='options-chain')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='volatility-surface')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        # Row 3: Technical indicators
        html.Div([
            dcc.Graph(id='technical-indicators')
        ], style={'width': '100%'}),
        
        # Row 4: Predictions and signals
        html.Div([
            html.Div([
                dcc.Graph(id='predictions-chart')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3("Recent Trading Signals", style={'textAlign': 'center'}),
                html.Div(id='signals-table')
            ], style={'width': '50%', 'display': 'inline-block', 'padding': 20})
        ])
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # 30 seconds
        n_intervals=0
    )
], style={'fontFamily': 'Arial, sans-serif', 'padding': 20})


@app.callback(
    [Output('price-chart', 'figure'),
     Output('greeks-gauge', 'figure'),
     Output('options-chain', 'figure'),
     Output('volatility-surface', 'figure'),
     Output('technical-indicators', 'figure'),
     Output('predictions-chart', 'figure'),
     Output('signals-table', 'children')],
    [Input('symbol-dropdown', 'value'),
     Input('timerange-dropdown', 'value'),
     Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(symbol, hours, n_clicks, n_intervals):
    """Update all dashboard components"""
    
    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    # Load data
    df_prices = timescale_db.query_price_history(symbol, start_time, end_time)
    df_indicators = timescale_db.query_technical_indicators(symbol, start_time, end_time)
    df_predictions = timescale_db.query_predictions(symbol, start_time, end_time)
    
    # 1. Price Chart
    price_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} Price', 'Volume')
    )
    
    if not df_prices.empty:
        # Candlestick chart
        price_fig.add_trace(
            go.Candlestick(
                x=df_prices['timestamp'],
                open=df_prices['open'],
                high=df_prices['high'],
                low=df_prices['low'],
                close=df_prices['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume bars
        price_fig.add_trace(
            go.Bar(
                x=df_prices['timestamp'],
                y=df_prices['volume'],
                name='Volume',
                marker_color='rgba(0, 150, 255, 0.5)'
            ),
            row=2, col=1
        )
    
    price_fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        title_text=f"{symbol} Price Action",
        template='plotly_white'
    )
    
    # 2. Greeks Gauge
greeks_fig = go.Figure()
    
    # Mock Greeks data (replace with actual query)
    greeks_data = {
        'Delta': 0.65,
        'Gamma': 0.03,
        'Vega': 0.25,
        'Theta': -0.05
    }
    
    for i, (greek, value) in enumerate(greeks_data.items()):
        greeks_fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': greek},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "lightgray"},
                    {'range': [-0.5, 0.5], 'color': "gray"},
                    {'range': [0.5, 1], 'color': "lightgray"}
                ]
            },
            domain={'row': i // 2, 'column': i % 2}
        ))
    
    greeks_fig.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=500,
        title_text="Options Greeks"
    )
    
    # 3. Options Chain
    options_fig = go.Figure()
    
    # Mock options data
    strikes = np.arange(150, 200, 5)
    call_prices = np.random.uniform(2, 30, len(strikes))
    put_prices = np.random.uniform(2, 30, len(strikes))
    
    options_fig.add_trace(go.Bar(
        x=strikes,
        y=call_prices,
        name='Call Options',
        marker_color='green'
    ))
    
    options_fig.add_trace(go.Bar(
        x=strikes,
        y=-put_prices,
        name='Put Options',
        marker_color='red'
    ))
    
    options_fig.update_layout(
        title=f"{symbol} Options Chain",
        xaxis_title="Strike Price",
        yaxis_title="Option Value",
        barmode='overlay',
        height=400,
        template='plotly_white'
    )
    
    # 4. Volatility Surface
    vol_fig = go.Figure()
    
    # Mock volatility surface
    strikes = np.arange(150, 200, 5)
    expirations = np.arange(1, 90, 5)
    X, Y = np.meshgrid(strikes, expirations)
    Z = 0.2 + 0.1 * np.sin(X / 20) + 0.05 * np.cos(Y / 10)
    
    vol_fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        name='IV Surface'
    ))
    
    vol_fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Days to Expiration',
            zaxis_title='IV'
        ),
        height=400
    )
    
    # 5. Technical Indicators
    tech_fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Moving Averages', 'RSI', 'MACD')
    )
    
    if not df_prices.empty and not df_indicators.empty:
        # Price and MAs
        tech_fig.add_trace(
            go.Scatter(x=df_prices['timestamp'], y=df_prices['close'],
                      name='Close', line=dict(color='black')),
            row=1, col=1
        )
        
        if 'sma_20' in df_indicators.columns:
            tech_fig.add_trace(
                go.Scatter(x=df_indicators['timestamp'], y=df_indicators['sma_20'],
                          name='SMA 20', line=dict(color='blue')),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in df_indicators.columns:
            tech_fig.add_trace(
                go.Scatter(x=df_indicators['timestamp'], y=df_indicators['rsi'],
                          name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            
            # Overbought/Oversold lines
            tech_fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            tech_fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'macd' in df_indicators.columns:
            tech_fig.add_trace(
                go.Scatter(x=df_indicators['timestamp'], y=df_indicators['macd'],
                          name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            
            if 'macd_signal' in df_indicators.columns:
                tech_fig.add_trace(
                    go.Scatter(x=df_indicators['timestamp'], y=df_indicators['macd_signal'],
                              name='Signal', line=dict(color='red')),
                    row=3, col=1
                )
    
    tech_fig.update_layout(height=600, template='plotly_white')
    
    # 6. Predictions Chart
    pred_fig = go.Figure()
    
    if not df_prices.empty:
        pred_fig.add_trace(go.Scatter(
            x=df_prices['timestamp'],
            y=df_prices['close'],
            name='Actual Price',
            line=dict(color='black', width=2)
        ))
    
    if not df_predictions.empty:
        pred_fig.add_trace(go.Scatter(
            x=df_predictions['timestamp'],
            y=df_predictions['predicted_price'],
            name='Predicted Price',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # Confidence intervals
        if 'confidence' in df_predictions.columns:
            upper = df_predictions['predicted_price'] * (1 + df_predictions['confidence'] * 0.1)
            lower = df_predictions['predicted_price'] * (1 - df_predictions['confidence'] * 0.1)
            
            pred_fig.add_trace(go.Scatter(
                x=df_predictions['timestamp'],
                y=upper,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,255,0)',
                showlegend=False
            ))
            
            pred_fig.add_trace(go.Scatter(
                x=df_predictions['timestamp'],
                y=lower,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,255,0)',
                fillcolor='rgba(0,0,255,0.2)',
                name='Confidence Interval'
            ))
    
    pred_fig.update_layout(
        title="Price Predictions",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400,
        template='plotly_white'
    )
    
    # 7. Signals Table
    signals_query = f"""
        SELECT timestamp, signal_type, confidence, target_price, stop_loss
        FROM trading_signals
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT 10
    """
    
    signals_html = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Time'),
                html.Th('Signal'),
                html.Th('Confidence'),
                html.Th('Target'),
                html.Th('Stop Loss')
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td('2024-01-15 10:30'),
                html.Td('BUY', style={'color': 'green', 'fontWeight': 'bold'}),
                html.Td('75%'),
                html.Td('$185.50'),
                html.Td('$178.00')
            ]),
            html.Tr([
                html.Td('2024-01-15 09:15'),
                html.Td('HOLD', style={'color': 'orange', 'fontWeight': 'bold'}),
                html.Td('62%'),
                html.Td('$180.00'),
                html.Td('$175.00')
            ])
        ])
    ], style={'width': '100%', 'border': '1px solid #ddd'})
    
    return price_fig, greeks_fig, options_fig, vol_fig, tech_fig, pred_fig, signals_html


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
