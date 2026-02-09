
from flask import Flask, render_template, request
from markupsafe import Markup
import pandas as pd
from pathlib import Path
import os

# Interactive charts
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

# ---------------- Configuration ----------------
CURRENCY_SYMBOL = "$"

# ---------------- Data Loading -----------------
HERE = Path(__file__).parent
CANDIDATES = [HERE / 'data' / 'superstore.csv', HERE / 'superstore.csv']

DATA_PATH = None
for p in CANDIDATES:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    env_path = Path(os.environ.get("SUPERSTORE_CSV", "")) if os.environ.get("SUPERSTORE_CSV") else None
    if env_path and env_path.exists():
        DATA_PATH = env_path

if DATA_PATH is None:
    raise SystemExit("Could not find superstore.csv. Put it in data/superstore.csv or ./superstore.csv, "
                     "or set SUPERSTORE_CSV to the path.")

df = pd.read_csv(DATA_PATH, encoding='latin1')

# --- Normalize columns (fix NBSPs, collapse/trim spaces) ---
df.columns = (
    df.columns
      .str.replace('\\u00a0', ' ', regex=False)   # NBSP → space
      .str.replace(r'\\s+', ' ', regex=True)      # collapse multiple spaces
      .str.strip()
)

# --- Robustly parse Order Date and create Year & Month ---
if 'Order Date' in df.columns:
    s = df['Order Date'].astype(str).str.replace('\\u00a0', ' ', regex=False).str.strip()
    d = pd.to_datetime(s, format='%m/%d/%Y', errors='coerce')
    if d.isna().mean() > 0.5:  # too many NaT? try looser inference
        d = pd.to_datetime(s, errors='coerce', infer_datetime_format=True, dayfirst=False)
    df['Order Date'] = d
    df['Year'] = d.dt.year
    df['Month'] = d.dt.to_period('M').astype(str)  # e.g., "2019-07"
    # --- Compute dataset min/max dates for UI constraints ---
    min_date = df['Order Date'].min()
    max_date = df['Order Date'].max()
else:
    df['Year'] = pd.NA
    df['Month'] = pd.NA
    min_date = None
    max_date = None

# --- Ensure numeric fields are numeric ---
for col in ['Sales', 'Profit', 'Discount']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Trim filter values to avoid space mismatches ---
for field in ['Category', 'Sub-Category', 'Region', 'Segment', 'State']:
    if field in df.columns:
        df[field] = (
            df[field]
              .astype(str)
              .str.replace('\\u00a0', ' ', regex=False)
              .str.strip()
              .replace({'nan': None})
        )

# ---------------- Dropdown Options ----------------
def unique_sorted(col):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())

FILTER_FIELDS = ['Category', 'Sub-Category', 'Region', 'Segment']
OPTIONS = {f: ['All'] + unique_sorted(f) for f in FILTER_FIELDS}

QUERY_OPTIONS = [
    ('dashboard', '📊 Dashboard'),
    ('total_sales_profit', 'Total Sales and Profit'),
    ('avg_discount_by_product', 'Average Discount by Product'),
    ('total_sales_by_year', 'Total Sales by Year'),
    ('profit_by_region', 'Profit by Region'),
    ('negative_profit_products', 'Products with Negative Profit'),
    ('top_customers_by_profit', 'Top Customers by Profit'),
    ('sales_by_category', 'Sales by Category'),
    ('monthly_sales_trend', 'Monthly Sales Trend'),
    ('top_products_sales', 'Top Products by Sales'),
    ('top_products_margin', 'Top Products by Profit Margin'),
    ('discount_impact', 'Discount Impact Analysis'),
    ('geomap_state_sales', 'Geo Map: Sales by State'),
]

# ---------------- Helpers ----------------
def apply_filters(base_df, selections, start_date=None, end_date=None):
    filtered = base_df.copy()
    # dropdowns
    for field in FILTER_FIELDS:
        val = selections.get(field, 'All')
        if val and val != 'All' and field in filtered.columns:
            left = (
                filtered[field]
                  .astype(str)
                  .str.replace('\\u00a0', ' ', regex=False)
                  .str.strip()
            )
            filtered = filtered[left == val]
    # date range
    if 'Order Date' in filtered.columns:
        if start_date:
            try:
                sd = pd.to_datetime(start_date)
                filtered = filtered[filtered['Order Date'] >= sd]
            except Exception:
                pass
        if end_date:
            try:
                ed = pd.to_datetime(end_date)
                filtered = filtered[filtered['Order Date'] <= ed]
            except Exception:
                pass
    return filtered

def format_currency(x):
    try:
        if pd.isna(x):
            return ""
        return f"{CURRENCY_SYMBOL}{x:,.2f}"
    except Exception:
        return str(x)

def format_currency_cols(df_in):
    df_out = df_in.copy()
    money_names = {'sales', 'profit', 'total sales', 'total profit', 'amount', 'revenue'}
    money_cols = [c for c in df_out.columns if c.lower() in money_names]
    for c in money_cols:
        df_out[c] = df_out[c].apply(format_currency)
    return df_out

# ------------- Geo helpers (US state name → abbreviation) -------------
STATE_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC',
    'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME',
    'Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV',
    'New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR',
    'Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

# ---------------- Query Functions ----------------
def run_total_sales_profit(d):
    out = pd.DataFrame({'Sales': [d['Sales'].sum()], 'Profit': [d['Profit'].sum()]})
    return 'Total Sales and Profit (Filtered)', out

def run_avg_discount_by_product(d):
    product_col = next((c for c in ['Product Name', 'Product', 'ProductName'] if c in d.columns), None)
    if not product_col:
        return 'Average Discount by Product', pd.DataFrame([{'Error': 'No product column found.'}])
    out = (
        d[[product_col, 'Discount']]
        .dropna(subset=['Discount'])
        .groupby(product_col, as_index=False)['Discount']
        .mean()
        .rename(columns={'Discount': 'Average Discount'})
        .sort_values('Average Discount', ascending=False)
    )
    return 'Average Discount by Product (Filtered)', out

def run_total_sales_by_year(d):
    if 'Year' not in d.columns or d['Year'].isna().all():
        return 'Total Sales by Year', pd.DataFrame([{'Error': 'No Year column found.'}])
    out = (
        d.groupby('Year', as_index=False)['Sales']
        .sum()
        .rename(columns={'Sales': 'Total Sales'})
        .sort_values('Year')
    )
    return 'Total Sales by Year (Filtered)', out

def run_profit_by_region(d):
    if 'Region' not in d.columns:
        return 'Profit by Region', pd.DataFrame([{'Error': 'No Region column found.'}])
    out = (
        d.groupby('Region', as_index=False)['Profit']
        .sum()
        .rename(columns={'Profit': 'Total Profit'})
        .sort_values('Total Profit', ascending=False)
    )
    return 'Profit by Region (Filtered)', out

def run_negative_profit_products(d):
    product_col = next((c for c in ['Product Name', 'Product', 'ProductName'] if c in d.columns), None)
    if not product_col:
        return 'Products with Negative Profit', pd.DataFrame([{'Error': 'No product column found.'}])
    out = d[d['Profit'] < 0][[product_col, 'Sales', 'Profit']].sort_values('Profit')
    return 'Products with Negative Profit (Filtered)', out

def run_top_customers_by_profit(d):
    cust_col = next((c for c in ['Customer Name', 'Customer', 'CustomerName'] if c in d.columns), None)
    if not cust_col:
        return 'Top Customers by Profit', pd.DataFrame([{'Error': 'No Customer column found.'}])
    out = (
        d.groupby(cust_col, as_index=False)['Profit']
        .sum()
        .sort_values('Profit', ascending=False)
        .head(10)
        .rename(columns={cust_col: 'Customer'})
    )
    return 'Top 10 Customers by Profit (Filtered)', out

def run_sales_by_category(d):
    if 'Category' not in d.columns:
        return 'Sales by Category', pd.DataFrame([{'Error': 'No Category column found.'}])
    out = (
        d.groupby('Category', as_index=False)['Sales']
        .sum()
        .rename(columns={'Sales': 'Total Sales'})
        .sort_values('Total Sales', ascending=False)
    )
    return 'Sales by Category (Filtered)', out

def run_monthly_sales_trend(d):
    if 'Month' not in d.columns:
        return 'Monthly Sales Trend', pd.DataFrame([{'Error': 'No Month column found.'}])
    out = (
        d.groupby('Month', as_index=False)['Sales']
        .sum()
        .rename(columns={'Sales': 'Total Sales'})
        .sort_values('Month')
    )
    return 'Monthly Sales Trend (Filtered)', out

def run_top_products_sales(d, k=15):
    product_col = next((c for c in ['Product Name', 'Product', 'ProductName'] if c in d.columns), None)
    if not product_col:
        return 'Top Products by Sales', pd.DataFrame([{'Error': 'No product column found.'}])
    out = (
        d.groupby(product_col, as_index=False)['Sales']
        .sum()
        .sort_values('Sales', ascending=False)
        .head(k)
        .rename(columns={product_col: 'Product', 'Sales': 'Total Sales'})
    )
    return f'Top {k} Products by Sales (Filtered)', out

def run_top_products_margin(d, k=15):
    product_col = next((c for c in ['Product Name', 'Product', 'ProductName'] if c in d.columns), None)
    if not product_col:
        return 'Top Products by Profit Margin', pd.DataFrame([{'Error': 'No product column found.'}])
    agg = d.groupby(product_col, as_index=False)[['Sales', 'Profit']].sum()
    agg['Margin %'] = (agg['Profit'] / agg['Sales']).replace([pd.NA, pd.NaT], 0)
    # Require a minimum sales threshold to avoid small denominators
    agg = agg[agg['Sales'] > 0]
    out = agg.sort_values('Margin %', ascending=False).head(k).rename(columns={product_col: 'Product'})
    return f'Top {k} Products by Profit Margin (Filtered)', out

def run_discount_impact(d):
    # Bin discounts and compute average margin
    if 'Discount' not in d.columns:
        return 'Discount Impact Analysis', pd.DataFrame([{'Error': 'No Discount column found.'}])
    work = d.copy()
    work['Margin %'] = work['Profit'] / work['Sales']
    bins = [-0.01, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    labels = ['0%', '0-5%', '5-10%', '10-20%', '20-50%', '50%+']
    work['Discount Band'] = pd.cut(work['Discount'].fillna(0), bins=bins, labels=labels)
    out = (
        work.groupby('Discount Band', as_index=False)
        .agg(**{
            'Transactions': ('Sales', 'count'),
            'Avg Discount': ('Discount', 'mean'),
            'Avg Margin %': ('Margin %', 'mean'),
            'Total Sales': ('Sales', 'sum'),
            'Total Profit': ('Profit', 'sum'),
        })
        .sort_values('Avg Discount')
    )
    return 'Discount Impact Analysis (Filtered)', out

def run_geomap_state_sales(d):
    if 'State' not in d.columns:
        return 'Geo Map: Sales by State', pd.DataFrame([{'Error': 'No State column found.'}])
    g = d.groupby('State', as_index=False)['Sales'].sum().rename(columns={'Sales':'Total Sales'})
    g['State Abbr'] = g['State'].map(STATE_ABBR).fillna('')
    g = g[g['State Abbr'] != '']  # keep only US states
    return 'Geo Map: Sales by US State (Filtered)', g

QUERY_RUNNERS = {
    'total_sales_profit': run_total_sales_profit,
    'avg_discount_by_product': run_avg_discount_by_product,
    'total_sales_by_year': run_total_sales_by_year,
    'profit_by_region': run_profit_by_region,
    'negative_profit_products': run_negative_profit_products,
    'top_customers_by_profit': run_top_customers_by_profit,
    'sales_by_category': run_sales_by_category,
    'monthly_sales_trend': run_monthly_sales_trend,
    'top_products_sales': run_top_products_sales,
    'top_products_margin': run_top_products_margin,
    'discount_impact': run_discount_impact,
    'geomap_state_sales': run_geomap_state_sales,
}

# ---------------- Chart Builder ----------------
def build_money_layout(fig):
    fig.update_layout(yaxis_tickprefix=CURRENCY_SYMBOL, yaxis_tickformat=",.2f")
    for tr in fig.data:
        if getattr(tr, "hovertemplate", None):
            continue
        tr.hovertemplate = f"%{{x}}: {CURRENCY_SYMBOL}%{{y:,.2f}}<extra></extra>"
    return fig

def to_plotly_html(fig):
    return Markup(fig.to_html(full_html=False, include_plotlyjs='cdn'))

def chart_for(key, out_df):
    fig = None
    if key == 'total_sales_profit':
        row = out_df.iloc[0] if not out_df.empty else pd.Series({'Sales': 0, 'Profit': 0})
        fig = go.Figure(data=[go.Bar(x=['Sales', 'Profit'], y=[row.get('Sales', 0), row.get('Profit', 0)])])
        fig.update_layout(title='Total Sales vs Profit', xaxis_title='', yaxis_title='Amount')
        fig = build_money_layout(fig)

    elif key == 'avg_discount_by_product':
        dfp = out_df.head(20)
        name_col = next((c for c in dfp.columns if c not in ['Average Discount'] and dfp[c].dtype == 'O'), None)
        x = dfp[name_col] if name_col else dfp.index.astype(str)
        fig = px.bar(dfp, x=x, y='Average Discount', title='Top 20 Products by Avg Discount')
        fig.update_layout(xaxis_title='Product', yaxis_tickformat=".1%")
        for tr in fig.data:
            tr.hovertemplate = "%{x}: %{y:.1%}<extra></extra>"

    elif key == 'total_sales_by_year':
        fig = px.line(out_df, x='Year', y='Total Sales', markers=True, title='Total Sales by Year')
        fig.update_layout(xaxis=dict(dtick=1))
        fig = build_money_layout(fig)

    elif key == 'profit_by_region':
        fig = px.bar(out_df, x='Region', y='Total Profit', title='Profit by Region')
        fig = build_money_layout(fig)

    elif key == 'negative_profit_products':
        dfp = out_df.nsmallest(30, 'Profit') if 'Profit' in out_df.columns else out_df.head(30)
        name_col = next((c for c in dfp.columns if c not in ['Sales', 'Profit'] and dfp[c].dtype == 'O'), None)
        x = dfp[name_col] if name_col else dfp.index.astype(str)
        fig = px.bar(dfp, x=x, y='Profit', title='Products with Negative Profit (Bottom 30)')
        fig.update_layout(xaxis_title='Product')
        fig = build_money_layout(fig)

    elif key == 'top_customers_by_profit':
        fig = px.bar(out_df, x='Customer', y='Profit', title='Top 10 Customers by Profit')
        fig = build_money_layout(fig)

    elif key == 'sales_by_category':
        fig = px.bar(out_df, x='Category', y='Total Sales', title='Sales by Category')
        fig = build_money_layout(fig)

    elif key == 'monthly_sales_trend':
        fig = px.line(out_df, x='Month', y='Total Sales', markers=True, title='Monthly Sales Trend')
        fig = build_money_layout(fig)

    elif key == 'top_products_sales':
        fig = px.bar(out_df, x='Product', y='Total Sales', title='Top Products by Sales')
        fig = build_money_layout(fig)

    elif key == 'top_products_margin':
        # Margin chart uses percentage on y
        fig = px.bar(out_df, x='Product', y='Margin %', title='Top Products by Profit Margin')
        fig.update_layout(yaxis_tickformat=".1%")
        for tr in fig.data:
            tr.hovertemplate = "%{x}: %{y:.1%}<extra></extra>"

    elif key == 'discount_impact':
        fig = px.bar(out_df, x='Discount Band', y='Avg Margin %', title='Discount Impact on Margin')
        fig.update_layout(yaxis_tickformat=".1%", xaxis_title='Discount Band')
        for tr in fig.data:
            tr.hovertemplate = "%{x}: %{y:.1%}<extra></extra>"

    elif key == 'geomap_state_sales':
        # Choropleth of US states by Total Sales
        if 'State Abbr' in out_df.columns:
            fig = px.choropleth(
                out_df, locations='State Abbr', color='Total Sales',
                locationmode='USA-states', scope='usa',
                title='Sales by US State', color_continuous_scale='Blues'
            )
            fig = build_money_layout(fig)

    if fig is None:
        return None
    return to_plotly_html(fig)

# ---------------- Dashboard Composer ----------------
def dashboard_components(d):
    cards = {}
    cards['Total Sales'] = format_currency(d['Sales'].sum())
    cards['Total Profit'] = format_currency(d['Profit'].sum())
    # Average order value (if we have Order ID / Row as proxy)
    denom = max(len(d), 1)
    cards['Avg Sales / Row'] = format_currency(d['Sales'].sum() / denom)

    # Build a few default dashboard charts
    title1, t1 = run_monthly_sales_trend(d)
    title2, t2 = run_sales_by_category(d)
    title3, t3 = run_profit_by_region(d)
    title4, t4 = run_top_products_sales(d, k=10)

    charts = [
        ('Monthly Sales Trend', chart_for('monthly_sales_trend', t1)),
        ('Sales by Category', chart_for('sales_by_category', t2)),
        ('Profit by Region', chart_for('profit_by_region', t3)),
        ('Top 10 Products by Sales', chart_for('top_products_sales', t4)),
    ]
    return cards, charts

# ---------------- Routes ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    selections = {f: request.form.get(f, 'All') for f in FILTER_FIELDS}
    selected_query = request.form.get('query', 'dashboard')
    start_date = request.form.get('start_date') or ''
    end_date = request.form.get('end_date') or ''
    result_title, result_html, graph_html = None, None, None

    if request.method == 'POST':
        filtered = apply_filters(df, selections, start_date, end_date)

        if selected_query == 'dashboard':
            cards, charts = dashboard_components(filtered)
            return render_template(
                'index.html',
                options=OPTIONS,
                query_options=QUERY_OPTIONS,
                selections=selections,
                selected_query=selected_query,
                result_title=None,
                result_html=None,
                graph_html=None,
                cards=cards,
                dashboard_charts=charts,
                start_date=start_date,
                end_date=end_date,
                min_date=min_date.strftime('%Y-%m-%d') if min_date is not None else None,
                max_date=max_date.strftime('%Y-%m-%d') if max_date is not None else None,
            )

        # normal single-query path
        runner = QUERY_RUNNERS.get(selected_query)
        if runner is None:
            result_title = "Unknown query"
            result_html = "<p>Unknown query selection.</p>"
        else:
            title, out_df = runner(filtered)
            result_title = title
            graph_html = chart_for(selected_query, out_df)
            display_df = format_currency_cols(out_df)
            result_html = display_df.to_html(classes='table table-striped', index=False, border=0, justify='center')

    return render_template(
        'index.html',
        options=OPTIONS,
        query_options=QUERY_OPTIONS,
        selections=selections,
        selected_query=selected_query,
        result_title=result_title,
        result_html=result_html,
        graph_html=graph_html,
        cards=None,
        dashboard_charts=None,
        start_date=start_date,
        end_date=end_date,
        min_date=min_date.strftime('%Y-%m-%d') if min_date is not None else None,
        max_date=max_date.strftime('%Y-%m-%d') if max_date is not None else None,
    )

if __name__ == '__main__':
    app.run(debug=True)
