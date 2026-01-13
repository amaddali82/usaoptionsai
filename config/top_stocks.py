"""
Top 50 Stocks by Sector/Category for Options Trading
Selected based on liquidity, volume, and market cap
"""

# Technology Sector - Top 50
TECHNOLOGY = [
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'ACN', 'AMD',
    'INTC', 'IBM', 'QCOM', 'TXN', 'INTU', 'NOW', 'MU', 'AMAT', 'ADI', 'LRCX',
    'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'PANW', 'CRWD', 'TEAM', 'WDAY', 'ZS',
    'DDOG', 'NET', 'SNOW', 'PLTR', 'DELL', 'HPQ', 'STX', 'WDC', 'NTAP', 'ANET',
    'AKAM', 'CFLT', 'DOCN', 'GTLB', 'MDB', 'OKTA', 'VEEV', 'ZM', 'TWLO', 'DOCU'
]

# Consumer Discretionary - Top 50
CONSUMER_DISCRETIONARY = [
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG',
    'ABNB', 'MAR', 'GM', 'F', 'ROST', 'YUM', 'DHI', 'LEN', 'HLT', 'DG',
    'ORLY', 'AZO', 'LULU', 'TPR', 'RL', 'ULTA', 'DPZ', 'QSR', 'RCL', 'CCL',
    'NCLH', 'LVS', 'WYNN', 'MGM', 'BBY', 'TGT', 'ETSY', 'W', 'CHWY', 'CVNA',
    'KMX', 'AN', 'LAD', 'GPC', 'AAP', 'TSCO', 'DKS', 'FIVE', 'OLLI', 'BURL'
]

# Financial Services - Top 50
FINANCIAL = [
    'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'SPGI', 'BLK',
    'C', 'SCHW', 'AXP', 'PGR', 'CB', 'MMC', 'ICE', 'CME', 'AON', 'TFC',
    'USB', 'PNC', 'COF', 'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'HIG',
    'AMP', 'BK', 'STT', 'NTRS', 'RF', 'CFG', 'KEY', 'FITB', 'HBAN', 'MTB',
    'ALLY', 'SYF', 'DFS', 'WRB', 'RJF', 'CINF', 'BRO', 'MKTX', 'LPLA', 'VOYA'
]

# Healthcare - Top 50
HEALTHCARE = [
    'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'REGN', 'VRTX', 'ISRG', 'ZTS', 'ELV', 'BIIB',
    'SYK', 'BSX', 'MDT', 'HCA', 'EW', 'IQV', 'BDX', 'RMD', 'IDXX', 'DXCM',
    'ALGN', 'HUM', 'CNC', 'MOH', 'MCK', 'CAH', 'COR', 'A', 'PODD', 'HOLX',
    'EXAS', 'TECH', 'INCY', 'NBIX', 'VTRS', 'OGN', 'PRGO', 'PEN', 'WAT', 'PKI'
]

# Communication Services - Top 50
COMMUNICATION = [
    'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
    'EA', 'TTWO', 'WBD', 'PARA', 'FOXA', 'FOX', 'LYV', 'NWSA', 'NWS', 'DISH',
    'OMC', 'IPG', 'MTCH', 'PINS', 'SNAP', 'SPOT', 'RBLX', 'U', 'ZI', 'TWTR',
    'NYT', 'ROKU', 'FUBO', 'MSG', 'MSGS', 'SIRI', 'LBRDA', 'LBRDK', 'WMG', 'SCOR',
    'IMAX', 'CNK', 'YELP', 'ANGI', 'CARS', 'TRIP', 'GRUB', 'ATUS', 'CABO', 'TGNA'
]

# Energy - Top 50
ENERGY = [
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB',
    'PXD', 'KMI', 'HES', 'BKR', 'HAL', 'DVN', 'FANG', 'TRGP', 'LNG', 'EQT',
    'MRO', 'APA', 'OKE', 'CTRA', 'CHRD', 'NOV', 'FTI', 'RIG', 'VAL', 'MTDR',
    'CNX', 'AR', 'MGY', 'CRGY', 'VTLE', 'PR', 'RRC', 'SM', 'CIVI', 'CLB',
    'HP', 'NFE', 'ENLC', 'WES', 'PAA', 'AM', 'EPD', 'ET', 'MPLX', 'PAGP'
]

# Industrials - Top 50
INDUSTRIALS = [
    'CAT', 'UNP', 'RTX', 'HON', 'BA', 'UPS', 'DE', 'LMT', 'GE', 'MMM',
    'NOC', 'GD', 'EMR', 'ETN', 'ITW', 'CSX', 'WM', 'NSC', 'FDX', 'CARR',
    'PCAR', 'JCI', 'TT', 'PH', 'ROK', 'FAST', 'OTIS', 'CPRT', 'CMI', 'AME',
    'ODFL', 'RSG', 'VRSK', 'IR', 'DAL', 'UAL', 'LUV', 'AAL', 'URI', 'PWR',
    'HWM', 'AXON', 'J', 'JBHT', 'CHRW', 'EXPD', 'XYL', 'DOV', 'FTV', 'IEX'
]

# Consumer Staples - Top 50
CONSUMER_STAPLES = [
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'GIS',
    'KMB', 'SYY', 'STZ', 'KHC', 'HSY', 'K', 'CHD', 'CLX', 'TSN', 'HRL',
    'CAG', 'SJM', 'CPB', 'MKC', 'LW', 'TAP', 'BG', 'ADM', 'DAR', 'INGR',
    'KR', 'SFM', 'GO', 'ACI', 'CASY', 'PSMT', 'SPB', 'JJSF', 'FLO', 'CALM',
    'EL', 'COTY', 'EPC', 'HAIN', 'THS', 'BGS', 'CENT', 'SMPL', 'KLG', 'FARM'
]

# Materials - Top 30 (smaller sector)
MATERIALS = [
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DD', 'DOW', 'NUE', 'VMC',
    'MLM', 'CTVA', 'ALB', 'BALL', 'PKG', 'AMCR', 'AVY', 'EMN', 'CE', 'FMC',
    'IFF', 'MOS', 'CF', 'PPG', 'RPM', 'SEE', 'WLK', 'AA', 'STLD', 'RS'
]

# Real Estate - Top 30
REAL_ESTATE = [
    'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'DLR', 'O', 'AVB', 'EQR',
    'SBAC', 'SPG', 'VICI', 'VTR', 'INVH', 'EXR', 'MAA', 'ESS', 'ARE', 'CBRE',
    'IRM', 'UDR', 'KIM', 'DOC', 'HST', 'CPT', 'REG', 'BXP', 'FRT', 'VNO'
]

# Utilities - Top 30
UTILITIES = [
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED',
    'WEC', 'EIX', 'ES', 'DTE', 'AWK', 'PPL', 'FE', 'ETR', 'AEE', 'CMS',
    'CNP', 'ATO', 'NI', 'LNT', 'PNW', 'OGE', 'EVRG', 'NWE', 'AVA', 'BKH'
]

# Combine all categories
ALL_SECTORS = {
    'Technology': TECHNOLOGY,
    'Consumer Discretionary': CONSUMER_DISCRETIONARY,
    'Financial': FINANCIAL,
    'Healthcare': HEALTHCARE,
    'Communication': COMMUNICATION,
    'Energy': ENERGY,
    'Industrials': INDUSTRIALS,
    'Consumer Staples': CONSUMER_STAPLES,
    'Materials': MATERIALS,
    'Real Estate': REAL_ESTATE,
    'Utilities': UTILITIES
}

# Flatten to get all unique symbols
ALL_SYMBOLS = list(set(
    TECHNOLOGY + CONSUMER_DISCRETIONARY + FINANCIAL + HEALTHCARE + 
    COMMUNICATION + ENERGY + INDUSTRIALS + CONSUMER_STAPLES + 
    MATERIALS + REAL_ESTATE + UTILITIES
))

# Top 50 most liquid for quick start
TOP_50_LIQUID = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK.B', 'JPM', 'V',
    'UNH', 'XOM', 'JNJ', 'MA', 'LLY', 'PG', 'AVGO', 'HD', 'CVX', 'MRK',
    'ABBV', 'COST', 'KO', 'PEP', 'WMT', 'ADBE', 'NFLX', 'CRM', 'TMO', 'BAC',
    'MCD', 'CSCO', 'ACN', 'NKE', 'DIS', 'LIN', 'AMD', 'DHR', 'ABT', 'PM',
    'TXN', 'INTC', 'ORCL', 'QCOM', 'VZ', 'CMCSA', 'WFC', 'NEE', 'UPS', 'IBM'
]

def get_symbols_by_sector(sector_name):
    """Get symbols for a specific sector"""
    return ALL_SECTORS.get(sector_name, [])

def get_all_symbols():
    """Get all unique symbols across all sectors"""
    return sorted(ALL_SYMBOLS)

def get_top_n_per_sector(n=10):
    """Get top N symbols from each sector"""
    result = {}
    for sector, symbols in ALL_SECTORS.items():
        result[sector] = symbols[:n]
    return result

if __name__ == '__main__':
    print(f"Total unique symbols: {len(ALL_SYMBOLS)}")
    print(f"\nSymbols per sector:")
    for sector, symbols in ALL_SECTORS.items():
        print(f"  {sector}: {len(symbols)}")
    print(f"\nTop 50 liquid stocks: {len(TOP_50_LIQUID)}")
