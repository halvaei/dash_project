import base64
import io

import requests
import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import ta  # تحلیل تکنیکال

# =================== توابع کمکی ===================

def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

def safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0

# دریافت داده بازار از TSETMC
def fetch_market_data():
    url = ("https://cdn.tsetmc.com/api/ClosingPrice/GetMarketWatch?market=0&"
           "paperTypes%5B0%5D=1&paperTypes%5B1%5D=2&paperTypes%5B2%5D=3&paperTypes%5B3%5D=4&"
           "paperTypes%5B4%5D=5&paperTypes%5B5%5D=6&paperTypes%5B6%5D=7&paperTypes%5B7%5D=8&"
           "paperTypes%5B8%5D=9&showTraded=false&withBestLimits=true")
    response = requests.get(url)
    data = response.json()
    records = []
    for row in data.get('marketwatch', []):
        try:
            insid = row.get('insID', '')
            if insid.startswith('IRB'):
                market = 'Bourse'
            elif insid.startswith('IRO'):
                market = 'Farabourse'
            else:
                market = 'Other'
            records.append({
                "Symbol": row.get('lva', ''),
                "Name": row.get('lvc', ''),
                "FinalPrice": safe_float(row.get('pcl', 0)),
                "LastPrice": safe_float(row.get('pf', 0)),
                "PercentChange": safe_float(row.get('pc', 0)),
                "Volume": safe_float(row.get('vc', 0)),
                "Value": safe_float(row.get('qtc', 0)),
                "Trades": safe_int(row.get('ztt', 0)),
                "Market": market
            })
        except:
            continue
    return pd.DataFrame(records)

def normalize(s):
    eps = 1e-9
    if s.max() == s.min():
        return s - s.min()
    return (s - s.min()) / (s.max() - s.min() + eps)

def normalize_weights(w_pc, w_vol, w_val):
    total = w_pc + w_vol + w_val
    if total == 0:
        return 0.5, 0.3, 0.2
    return w_pc / total, w_vol / total, w_val / total

def compute_composite_score(df, w_pc=0.5, w_vol=0.3, w_val=0.2):
    df = df.copy()
    df['norm_pc'] = normalize(df['PercentChange'])
    df['norm_vol'] = normalize(df['Volume'])
    df['norm_val'] = normalize(df['Value'])
    df['CompositeScore'] = w_pc * df['norm_pc'] + w_vol * df['norm_vol'] + w_val * df['norm_val']
    return df

def compute_dynamic_thresholds(df):
    positive_pc = df[df['PercentChange'] > 0]['PercentChange']
    volume = df['Volume']
    pc_threshold = positive_pc.quantile(0.6) if not positive_pc.empty else 3.0
    vol_threshold = volume.quantile(0.5) if not volume.empty else 10000
    return round(pc_threshold, 2), int(vol_threshold)

# اندیکاتورهای تحلیل تکنیکال: SMA, EMA, RSI, MACD
def add_technical_indicators(df):
    df = df.copy()
    df['SMA_5'] = ta.trend.sma_indicator(df['<CLOSE>'], window=5)
    df['SMA_10'] = ta.trend.sma_indicator(df['<CLOSE>'], window=10)
    df['EMA_10'] = ta.trend.ema_indicator(df['<CLOSE>'], window=10)
    df['RSI_14'] = ta.momentum.rsi(df['<CLOSE>'], window=14)
    macd = ta.trend.MACD(df['<CLOSE>'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(df['<CLOSE>'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    return df

# افزودن ایچی‌موکو
def add_ichimoku(df):
    high_9 = df['<HIGH>'].rolling(window=9).max()
    low_9 = df['<LOW>'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['<HIGH>'].rolling(window=26).max()
    low_26 = df['<LOW>'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    high_52 = df['<HIGH>'].rolling(window=52).max()
    low_52 = df['<LOW>'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    df['chikou_span'] = df['<CLOSE>'].shift(-26)
    return df

# افزودن سطوح فیبوناچی
def add_fibonacci_levels(df):
    max_price = df['<HIGH>'].max()
    min_price = df['<LOW>'].min()
    diff = max_price - min_price
    levels = {
        "0%": max_price,
        "23.6%": max_price - 0.236 * diff,
        "38.2%": max_price - 0.382 * diff,
        "50%": max_price - 0.5 * diff,
        "61.8%": max_price - 0.618 * diff,
        "78.6%": max_price - 0.786 * diff,
        "100%": min_price
    }
    return levels

# تولید سیگنال معاملاتی (SMA, RSI, MACD)
def get_trade_signal(df):
    if len(df) < 2:
        return "داده کافی برای تولید سیگنال وجود ندارد."
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signals = []

    if prev['SMA_5'] <= prev['SMA_10'] and latest['SMA_5'] > latest['SMA_10']:
        signals.append("سیگنال خرید: SMA 5 از SMA 10 به بالا عبور کرد.")
    elif prev['SMA_5'] >= prev['SMA_10'] and latest['SMA_5'] < latest['SMA_10']:
        signals.append("سیگنال فروش: SMA 5 از SMA 10 به پایین عبور کرد.")

    if latest['RSI_14'] is not None:
        if latest['RSI_14'] > 65:
            signals.append("RSI بالای 65: اشباع خرید.")
        elif latest['RSI_14'] < 35:
            signals.append("RSI زیر 35: اشباع فروش.")

    if prev['MACD'] <= prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
        signals.append("سیگنال خرید MACD.")
    elif prev['MACD'] >= prev['MACD_signal'] and latest['MACD'] < latest['MACD_signal']:
        signals.append("سیگنال فروش MACD.")

    if not signals:
        return "هیچ سیگنال خاصی یافت نشد."
    return " - ".join(signals)

# سیگنال ایچی‌موکو
def ichimoku_signal(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price = latest['<CLOSE>']
    span_a = latest['senkou_span_a']
    span_b = latest['senkou_span_b']
    tenkan = latest['tenkan_sen']
    kijun = latest['kijun_sen']
    prev_tenkan = previous['tenkan_sen']
    prev_kijun = previous['kijun_sen']

    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    signals = []

    if price > cloud_top:
        signals.append("قیمت بالای ابر ایچی‌موکو است: روند صعودی")
    elif price < cloud_bottom:
        signals.append("قیمت پایین ابر ایچی‌موکو است: روند نزولی")
    else:
        signals.append("قیمت داخل ابر ایچی‌موکو است: بازار در حالت خنثی")

    if prev_tenkan < prev_kijun and tenkan > kijun:
        signals.append("کراس صعودی تنکان-کجون: سیگنال خرید")
    elif prev_tenkan > prev_kijun and tenkan < kijun:
        signals.append("کراس نزولی تنکان-کجون: سیگنال فروش")

    return " - ".join(signals)

# تابع تصمیم‌گیری منطقی بر اساس اندیکاتورها
def decision_logic(df):
    if len(df) < 2:
        return "داده کافی برای تصمیم‌گیری وجود ندارد."

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = latest['<CLOSE>']
    tenkan = latest['tenkan_sen']
    kijun = latest['kijun_sen']
    span_a = latest['senkou_span_a']
    span_b = latest['senkou_span_b']
    rsi = latest['RSI_14']
    macd = latest['MACD']
    macd_signal = latest['MACD_signal']
    sma5 = latest['SMA_5']
    sma10 = latest['SMA_10']

    # وضعیت روند ایچی‌موکو
    if price > max(span_a, span_b):
        ichi_trend = 'صعودی'
    elif price < min(span_a, span_b):
        ichi_trend = 'نزولی'
    else:
        ichi_trend = 'خنثی'

    # کراس SMA
    if prev['SMA_5'] < prev['SMA_10'] and sma5 > sma10:
        sma_signal = 'کراس صعودی'
    elif prev['SMA_5'] > prev['SMA_10'] and sma5 < sma10:
        sma_signal = 'کراس نزولی'
    else:
        sma_signal = 'خنثی'

    # MACD
    if macd > macd_signal:
        macd_signal_state = 'صعودی'
    else:
        macd_signal_state = 'نزولی'

    # تصمیم‌گیری
    if ichi_trend == 'نزولی' and rsi < 35 and macd_signal_state == 'نزولی' and sma_signal == 'کراس نزولی':
        return "روند نزولی قوی؛ پیشنهاد: فروش یا حفظ موقعیت فروش."
    elif ichi_trend == 'صعودی' and rsi > 45 and macd_signal_state == 'صعودی' and sma_signal == 'کراس صعودی':
        return "روند صعودی؛ پیشنهاد: خرید یا حفظ موقعیت خرید."
    else:
        return "بازار نامشخص؛ صبر کنید تا سیگنال‌های قوی‌تری ظاهر شود."

# تبدیل فایل بارگذاری‌شده به DataFrame
def parse_uploaded_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, "فرمت فایل پشتیبانی نمی‌شود."
        return df, None
    except Exception as e:
        return None, f"خطا در خواندن فایل: {str(e)}"

# =================== اپلیکیشن Dash ===================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "داشبورد بورس با امتیاز ترکیبی دینامیک و تحلیل تکنیکال"

app.layout = dbc.Container([
    html.H2("📊 داشبورد تحلیل بورس با امتیاز ترکیبی دینامیک", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dcc.RadioItems(
                id="mode-selector",
                options=[
                    {"label": "حالت ثابت (پیش‌فرض)", "value": "static"},
                    {"label": "حالت دینامیک (قابل تنظیم)", "value": "dynamic"}
                ],
                value="static",
                labelStyle={"display": "block"}
            )
        ], width=3),

        dbc.Col([
            html.Label("درصد تغییر (وزن):"),
            dcc.Slider(id='weight-pc', min=0, max=1, step=0.05, value=0.5,
                       marks={0: '0', 0.5: '0.5', 1: '1'}, tooltip={"always_visible": True}),
            html.Label("حجم معاملات (وزن):"),
            dcc.Slider(id='weight-vol', min=0, max=1, step=0.05, value=0.3,
                       marks={0: '0', 0.5: '0.5', 1: '1'}, tooltip={"always_visible": True}),
            html.Label("ارزش معاملات (وزن):"),
            dcc.Slider(id='weight-val', min=0, max=1, step=0.05, value=0.2,
                       marks={0: '0', 0.5: '0.5', 1: '1'}, tooltip={"always_visible": True}),
        ], width=6),

        dbc.Col([
            dcc.Dropdown(
                id='market-selection',
                options=[
                    {"label": "کل بازار", "value": "all"},
                    {"label": "بورس", "value": "Bourse"},
                    {"label": "فرابورس", "value": "Farabourse"},
                ],
                value="all",
                clearable=False,
            ),
            dcc.Input(id='min-volume-input', type='number', min=0, value=10000,
                      placeholder='حداقل حجم', className="mt-2"),
            dcc.Slider(
                id='percent-change-slider',
                min=-50, max=50, step=0.5,
                marks={i: f"{i}%" for i in range(-50, 51, 10)},
                value=3, tooltip={"always_visible": True}
            ),
            dbc.Checklist(
                options=[{"label": "فیلتر داینامیک", "value": "dynamic"}],
                value=["dynamic"],
                id="use-dynamic-filter",
                switch=True,
                className="mt-2"
            )
        ], width=3)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='top-growth-chart'))
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='market-kpis', className='mb-4', style={"fontWeight": "bold", "fontSize": "18px"}))
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='stocks-table',
                columns=[
                    {"name": "نماد", "id": "Symbol"},
                    {"name": "درصد تغییر", "id": "PercentChange", "type": "numeric"},
                    {"name": "حجم", "id": "Volume", "type": "numeric"},
                    {"name": "ارزش", "id": "Value", "type": "numeric"},
                    {"name": "بازار", "id": "Market"},
                    {"name": "امتیاز", "id": "CompositeScore", "type": "numeric", "format": {"specifier": ".3f"}},
                ],
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_table={'overflowX': 'auto'},
                style_cell={"textAlign": "center", 'fontFamily': 'IRANSans'},
                style_header={'backgroundColor': '#0d6efd', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'PercentChange', 'filter_query': '{PercentChange} > 0'},
                        'color': 'green',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'PercentChange', 'filter_query': '{PercentChange} < 0'},
                        'color': 'red',
                        'fontWeight': 'bold'
                    }
                ]
            )
        ])
    ]),

    html.Hr(),

    html.H4("📂 بارگذاری فایل اکسل داده‌های کندل روزانه"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'فایل اکسل خود را اینجا بکشید یا ',
            html.A('انتخاب کنید')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px 0'
        },
        multiple=False
    ),

    dbc.Row([
        dbc.Col(dcc.Graph(id='candlestick-chart'))
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='trade-signals', style={"fontWeight": "bold", "fontSize": "18px", "color": "darkblue"}))
    ])

], fluid=True)


# =================== Callback ها ===================

@app.callback(
    Output('top-growth-chart', 'figure'),
    Output('stocks-table', 'data'),
    Output('market-kpis', 'children'),
    Input('mode-selector', 'value'),
    Input('weight-pc', 'value'),
    Input('weight-vol', 'value'),
    Input('weight-val', 'value'),
    Input('market-selection', 'value'),
    Input('min-volume-input', 'value'),
    Input('percent-change-slider', 'value'),
    Input('use-dynamic-filter', 'value'),
)
def update_dashboard(mode, w_pc, w_vol, w_val, market, min_vol, min_pc, dynamic_filter):
    df = fetch_market_data()

    # فیلتر بازار
    if market != "all":
        df = df[df['Market'] == market]

    # فیلتر حجم و درصد تغییر
    use_dynamic = "dynamic" in dynamic_filter
    if use_dynamic:
        pc_threshold, vol_threshold = compute_dynamic_thresholds(df)
        min_pc = max(pc_threshold, min_pc)
        min_vol = max(vol_threshold, min_vol)

    df_filtered = df[(df['PercentChange'] > min_pc) & (df['Volume'] > min_vol)]

    # محاسبه امتیاز ترکیبی
    if mode == "static":
        w1, w2, w3 = 0.5, 0.3, 0.2
    else:
        w1, w2, w3 = normalize_weights(w_pc, w_vol, w_val)

    df_scored = compute_composite_score(df_filtered, w1, w2, w3)
    df_scored = df_scored.sort_values(by='CompositeScore', ascending=False)

    top10 = df_scored.head(10)

    fig = px.bar(
        top10,
        x='Symbol',
        y='CompositeScore',
        color='PercentChange',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='📈 ۱۰ نماد برتر با بالاترین امتیاز ترکیبی',
        labels={'Symbol': 'نماد', 'CompositeScore': 'امتیاز ترکیبی'}
    )
    fig.update_layout(template='plotly_white')

    kpis = [
        f"تعداد کل نمادها: {len(df):,}",
        f"تعداد نمادهای بالاتر از فیلتر: {len(df_filtered):,}",
        f"آستانه درصد تغییر: {min_pc}%",
        f"آستانه حجم معاملات: {min_vol:,}"
    ]
    kpi_text = " | ".join(kpis)

    return fig, df_scored.to_dict('records'), kpi_text


@app.callback(
    Output('candlestick-chart', 'figure'),
    Output('trade-signals', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def update_candlestick(contents, filename):
    if contents is None:
        fig = go.Figure()
        fig.update_layout(
            title="نمودار کندل استیک اینجا نمایش داده می‌شود پس از بارگذاری فایل اکسل",
            xaxis_title="تاریخ",
            yaxis_title="قیمت",
            template="plotly_white"
        )
        return fig, ""

    df, error = parse_uploaded_file(contents, filename)
    if error:
        fig = go.Figure()
        fig.update_layout(title=f"❌ {error}")
        return fig, ""

    required_cols = ['<DTYYYYMMDD>', '<FIRST>', '<HIGH>', '<LOW>', '<CLOSE>']
    if not all(col in df.columns for col in required_cols):
        fig = go.Figure()
        fig.update_layout(title=f"❌ فایل بارگذاری شده ستون‌های لازم را ندارد.")
        return fig, ""

    df['Date'] = pd.to_datetime(df['<DTYYYYMMDD>'], format='%Y%m%d')
    df = df.sort_values('Date')

    df = add_technical_indicators(df)
    df = add_ichimoku(df)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.5, 0.2, 0.3],
        subplot_titles=('نمودار کندل استیک و میانگین‌های متحرک', 'ابر ایچی‌موکو', 'RSI')
    )

    # کندل استیک
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['<FIRST>'], high=df['<HIGH>'], low=df['<LOW>'], close=df['<CLOSE>'],
        increasing_line_color='green', decreasing_line_color='red', name='کندل'
    ), row=1, col=1)

    # SMA و EMA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'], mode='lines', name='SMA 5', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_10'], mode='lines', name='SMA 10', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_10'], mode='lines', name='EMA 10', line=dict(color='purple')), row=1, col=1)

    # خطوط فیبوناچی
    fib_levels = add_fibonacci_levels(df)
    for level_name, price in fib_levels.items():
        fig.add_hline(y=price, line_dash="dot", line_color="gray",
                      annotation_text=f"Fib {level_name}", annotation_position="top left", row=1, col=1)

    # ایچی‌موکو
    fig.add_trace(go.Scatter(x=df['Date'], y=df['tenkan_sen'], mode='lines', name='Tenkan-sen', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['kijun_sen'], mode='lines', name='Kijun-sen', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['senkou_span_a'], fill=None, mode='lines', line=dict(color='lightgreen'), name='Senkou Span A'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['senkou_span_b'], fill='tonexty', mode='lines', line=dict(color='lightcoral'), name='Senkou Span B'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], mode='lines', name='RSI', line=dict(color='black')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        title=f"نمودار کندل استیک همراه با ایچی‌موکو و فیبوناچی - فایل: {filename}",
        height=900,
    )

    ich_signal = ichimoku_signal(df)
    trade_signal_text = get_trade_signal(df)
    decision = decision_logic(df)

    full_signal = ich_signal + " | " + trade_signal_text + " | " + decision

    return fig, full_signal


if __name__ == '__main__':
    app.run(debug=True)

