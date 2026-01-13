"""
Simplified Dash Dashboard - Can run standalone without Docker
"""
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Dash app
app = dash.Dash(__name__, title="USA Options AI Dashboard")
server = app.server

# Generate sample data for demonstration
def generate_sample_data():
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
    
    # Sample stock price data
    base_price = 180
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2)
    
    df_prices = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(len(dates)) * 0.5,
        'high': prices + np.random.randn(len(dates)) * 0.5 + 1,
        'low': prices + np.random.randn(len(dates)) * 0.5 - 1,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Sample technical indicators
    df_prices['sma_20'] = df_prices['close'].rolling(20).mean()
    df_prices['rsi'] = 50 + np.random.randn(len(dates)) * 20
    df_prices['rsi'] = df_prices['rsi'].clip(0, 100)
    
    return df_prices

# Define layout
app.layout = html.Div([
    html.Div([
        html.H1("🚀 USA Options AI - Real-time Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        html.H3("Intelligent Stock Options Prediction System", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Select Symbol:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[
                    {'label': '🍎 AAPL - Apple Inc.', 'value': 'AAPL'},
                    {'label': '🪟 MSFT - Microsoft', 'value': 'MSFT'},
                    {'label': '🔍 GOOGL - Alphabet', 'value': 'GOOGL'},
                    {'label': '📦 AMZN - Amazon', 'value': 'AMZN'},
                    {'label': '⚡ TSLA - Tesla', 'value': 'TSLA'}
                ],
                value='AAPL',
                style={'width': '250px'}
            )
        ], style={'display': 'inline-block', 'marginRight': 20}),
        
        html.Div([
            html.Label("Time Range:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='timerange-dropdown',
                options=[
                    {'label': '1 Hour', 'value': 1},
                    {'label': '1 Day', 'value': 24},
                    {'label': '1 Week', 'value': 168},
                    {'label': '1 Month', 'value': 720}
                ],
                value=720,
                style={'width': '150px'}
            )
        ], style={'display': 'inline-block', 'marginRight': 20}),
        
        html.Button('🔄 Refresh Data', id='refresh-button', n_clicks=0,
                   style={
                       'marginTop': 20,
                       'padding': '10px 20px',
                       'backgroundColor': '#3498db',
                       'color': 'white',
                       'border': 'none',
                       'borderRadius': '5px',
                       'cursor': 'pointer',
                       'fontSize': '14px',
                       'fontWeight': 'bold'
                   })
    ], style={'marginBottom': 30, 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': '10px'}),
    
    # Status indicators
    html.Div([
        html.Div([
            html.H3("💰 Current Price", style={'color': '#27ae60'}),
            html.H2(id='current-price', style={'color': '#2c3e50'})
        ], style={'flex': 1, 'textAlign': 'center', 'padding': 20, 'backgroundColor': '#e8f8f5', 'borderRadius': '10px', 'marginRight': 10}),
        
        html.Div([
            html.H3("📈 24h Change", style={'color': '#3498db'}),
            html.H2(id='price-change', style={'color': '#2c3e50'})
        ], style={'flex': 1, 'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ebf5fb', 'borderRadius': '10px', 'marginRight': 10}),
        
        html.Div([
            html.H3("🎯 Signal", style={'color': '#9b59b6'}),
            html.H2(id='trading-signal', style={'color': '#2c3e50'})
        ], style={'flex': 1, 'textAlign': 'center', 'padding': 20, 'backgroundColor': '#f4ecf7', 'borderRadius': '10px', 'marginRight': 10}),
        
        html.Div([
            html.H3("🔮 Confidence", style={'color': '#e74c3c'}),
            html.H2(id='confidence-level', style={'color': '#2c3e50'})
        ], style={'flex': 1, 'textAlign': 'center', 'padding': 20, 'backgroundColor': '#fadbd8', 'borderRadius': '10px'})
    ], style={'display': 'flex', 'marginBottom': 30}),
    
    # Main charts
    html.Div([
        html.Div([
            dcc.Graph(id='price-chart', style={'height': '500px'})
        ], style={'width': '100%', 'marginBottom': 20}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='rsi-chart', style={'height': '300px'})
            ], style={'width': '50%', 'display': 'inline-block', 'paddingRight': 10}),
            
            html.Div([
                dcc.Graph(id='volume-chart', style={'height': '300px'})
            ], style={'width': '50%', 'display': 'inline-block', 'paddingLeft': 10})
        ]),
        
        # Trading signals table
        html.Div([
            html.H3("📊 Recent Trading Signals", style={'textAlign': 'center', 'marginTop': 30}),
            html.Div(id='signals-display', style={'padding': 20})
        ], style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'marginTop': 30})
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # 60 seconds
        n_intervals=0
    )
], style={'fontFamily': 'Arial, sans-serif', 'padding': 30, 'backgroundColor': '#ffffff'})


@app.callback(
    [Output('current-price', 'children'),
     Output('price-change', 'children'),
     Output('trading-signal', 'children'),
     Output('confidence-level', 'children'),
     Output('price-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('signals-display', 'children')],
    [Input('symbol-dropdown', 'value'),
     Input('timerange-dropdown', 'value'),
     Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(symbol, hours, n_clicks, n_intervals):
    """Update all dashboard components"""
    
    # Generate sample data
    df = generate_sample_data()
    
    # Calculate metrics
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-24] if len(df) >= 24 else df['close'].iloc[0]
    price_change_pct = ((current_price - previous_price) / previous_price) * 100
    
    # Simulated trading signal
    signals = ['BUY', 'SELL', 'HOLD', 'STRONG BUY']
    signal = np.random.choice(signals)
    confidence = np.random.randint(65, 95)
    
    # Status card values
    price_display = f"${current_price:.2f}"
    change_display = f"{price_change_pct:+.2f}%"
    change_color = '#27ae60' if price_change_pct >= 0 else '#e74c3c'
    signal_display = signal
    confidence_display = f"{confidence}%"
    
    # Price Chart
    price_fig = go.Figure()
    
    price_fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    price_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['sma_20'],
        name='SMA 20',
        line=dict(color='orange', width=2)
    ))
    
    price_fig.update_layout(
        title=f"{symbol} Price Chart with Moving Average",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    # RSI Chart
    rsi_fig = go.Figure()
    
    rsi_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['rsi'],
        name='RSI',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        fillcolor='rgba(147, 51, 234, 0.1)'
    ))
    
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    rsi_fig.update_layout(
        title="RSI (Relative Strength Index)",
        xaxis_title="Time",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        template='plotly_white'
    )
    
    # Volume Chart
    volume_fig = go.Figure()
    
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
              for i in range(len(df))]
    
    volume_fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color=colors
    ))
    
    volume_fig.update_layout(
        title="Trading Volume",
        xaxis_title="Time",
        yaxis_title="Volume",
        template='plotly_white'
    )
    
    # Trading signals table
    signals_data = [
        {'time': '2 min ago', 'signal': 'BUY', 'price': '$182.50', 'confidence': '78%', 'status': '✅ Active'},
        {'time': '15 min ago', 'signal': 'HOLD', 'price': '$181.20', 'confidence': '65%', 'status': '⏸️ Closed'},
        {'time': '1 hour ago', 'signal': 'STRONG BUY', 'price': '$180.00', 'confidence': '85%', 'status': '✅ Executed'},
        {'time': '3 hours ago', 'signal': 'SELL', 'price': '$183.50', 'confidence': '72%', 'status': '⏸️ Closed'}
    ]
    
    signals_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Time', style={'padding': '10px', 'textAlign': 'left'}),
                html.Th('Signal', style={'padding': '10px', 'textAlign': 'left'}),
                html.Th('Price', style={'padding': '10px', 'textAlign': 'left'}),
                html.Th('Confidence', style={'padding': '10px', 'textAlign': 'left'}),
                html.Th('Status', style={'padding': '10px', 'textAlign': 'left'})
            ], style={'backgroundColor': '#3498db', 'color': 'white'})
        ),
        html.Tbody([
            html.Tr([
                html.Td(row['time'], style={'padding': '10px'}),
                html.Td(row['signal'], style={
                    'padding': '10px',
                    'fontWeight': 'bold',
                    'color': '#27ae60' if 'BUY' in row['signal'] else '#e74c3c' if row['signal'] == 'SELL' else '#f39c12'
                }),
                html.Td(row['price'], style={'padding': '10px'}),
                html.Td(row['confidence'], style={'padding': '10px'}),
                html.Td(row['status'], style={'padding': '10px'})
            ], style={'backgroundColor': '#f8f9fa' if i % 2 == 0 else 'white'})
            for i, row in enumerate(signals_data)
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
    return (
        price_display,
        html.Span(change_display, style={'color': change_color, 'fontWeight': 'bold'}),
        html.Span(signal_display, style={
            'color': '#27ae60' if 'BUY' in signal else '#e74c3c' if signal == 'SELL' else '#f39c12',
            'fontWeight': 'bold'
        }),
        confidence_display,
        price_fig,
        rsi_fig,
        volume_fig,
        signals_table
    )


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 USA Options AI Dashboard Starting...")
    print("="*70)
    print("\n📊 Dashboard URL: http://localhost:8050")
    print("🔄 Auto-refresh: Every 60 seconds")
    print("\n💡 Features:")
    print("  ✓ Real-time price charts with candlesticks")
    print("  ✓ Technical indicators (RSI, Moving Averages)")
    print("  ✓ Trading volume analysis")
    print("  ✓ AI-powered trading signals")
    print("  ✓ Interactive controls and filters")
    print("\n⚠️  Note: Currently showing sample data for demonstration")
    print("   Connect to live databases to see real-time data")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8050)
