
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Global style ─────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi': 150,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})
COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2',
          '#937860','#DA8BC3','#8C8C8C']

# ════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 1 : LOADING DATA")
print("="*55)

df = pd.read_csv('ecommerce_sales_data.csv', parse_dates=['Date'])
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {df.shape[1]}")
print(f"\nFirst 5 rows:\n{df.head()}\n")

# ════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ════════════════════════════════════════════════════════════
print("="*55)
print("  STEP 2 : DATA CLEANING")
print("="*55)

print(f"  Missing values:\n{df.isnull().sum()}")
print(f"\n  Duplicate rows : {df.duplicated().sum()}")
print(f"  Data types:\n{df.dtypes}")

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)

# Derived columns
df['Month_Num'] = df['Date'].dt.month
df['Revenue']   = df['Final_Amount']   # alias for clarity

print("\n  ✅ Data cleaning complete — no issues found.")

# ════════════════════════════════════════════════════════════
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 3 : EXPLORATORY DATA ANALYSIS")
print("="*55)

total_revenue   = df['Revenue'].sum()
total_orders    = len(df)
avg_order_value = df['Revenue'].mean()
total_qty       = df['Quantity'].sum()
delivered_pct   = (df['Order_Status'] == 'Delivered').mean() * 100

print(f"\n  📦 Total Orders       : {total_orders:,}")
print(f"  💰 Total Revenue      : ₹{total_revenue:,.0f}")
print(f"  🛒 Avg Order Value    : ₹{avg_order_value:,.0f}")
print(f"  📬 Delivery Rate      : {delivered_pct:.1f}%")
print(f"  🏷️  Total Items Sold   : {total_qty:,}")

# ── Category analysis ────────────────────────────────────────
cat_rev  = df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
cat_ord  = df.groupby('Category')['Order_ID'].count().sort_values(ascending=False)
print(f"\n  Top category by revenue : {cat_rev.index[0]}  (₹{cat_rev.iloc[0]:,.0f})")
print(f"  Top category by orders  : {cat_ord.index[0]}  ({cat_ord.iloc[0]} orders)")

# ── City analysis ─────────────────────────────────────────────
city_rev = df.groupby('City')['Revenue'].sum().sort_values(ascending=False)
print(f"\n  Top city by revenue     : {city_rev.index[0]}  (₹{city_rev.iloc[0]:,.0f})")

# ── Monthly revenue ───────────────────────────────────────────
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
monthly_rev = (df.groupby('Month')['Revenue']
                 .sum()
                 .reindex(month_order)
                 .dropna())

best_month  = monthly_rev.idxmax()
print(f"\n  Best sales month        : {best_month}  (₹{monthly_rev[best_month]:,.0f})")

# ── Payment method ─────────────────────────────────────────────
pay_counts = df['Payment_Method'].value_counts()
print(f"\n  Most used payment       : {pay_counts.index[0]}  ({pay_counts.iloc[0]} orders)")

# ════════════════════════════════════════════════════════════
# STEP 4 — VISUALISATIONS  (8 charts saved to PNG)
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 4 : CREATING VISUALISATIONS")
print("="*55)

fig, axes = plt.subplots(4, 2, figsize=(16, 22))
fig.suptitle('E-Commerce Sales Analysis Dashboard — 2023\nAnalyst: Sakthiganesh K',
             fontsize=16, fontweight='bold', y=0.98)
axes = axes.flatten()

# ── Chart 1 : Revenue by Category (bar) ──────────────────────
ax = axes[0]
bars = ax.bar(cat_rev.index, cat_rev.values, color=COLORS, edgecolor='white', linewidth=0.5)
ax.set_title('Revenue by Category')
ax.set_xlabel('Category')
ax.set_ylabel('Revenue (₹)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=30)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
            f'₹{bar.get_height()/1e6:.1f}M', ha='center', va='bottom', fontsize=7)

# ── Chart 2 : Monthly Revenue Trend (line) ───────────────────
ax = axes[1]
ax.plot(range(len(monthly_rev)), monthly_rev.values,
        marker='o', color='#4C72B0', linewidth=2.5, markersize=7)
ax.fill_between(range(len(monthly_rev)), monthly_rev.values, alpha=0.15, color='#4C72B0')
ax.set_title('Monthly Revenue Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue (₹)')
ax.set_xticks(range(len(monthly_rev)))
ax.set_xticklabels([m[:3] for m in monthly_rev.index], rotation=30)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))

# ── Chart 3 : Top 10 Cities by Revenue (horizontal bar) ──────
ax = axes[2]
top_cities = city_rev.head(10)
ax.barh(top_cities.index[::-1], top_cities.values[::-1], color='#55A868', edgecolor='white')
ax.set_title('Top 10 Cities by Revenue')
ax.set_xlabel('Revenue (₹)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))
for i, (val, name) in enumerate(zip(top_cities.values[::-1], top_cities.index[::-1])):
    ax.text(val + 10000, i, f'₹{val/1e6:.1f}M', va='center', fontsize=7)

# ── Chart 4 : Payment Method Distribution (pie) ───────────────
ax = axes[3]
ax.pie(pay_counts.values, labels=pay_counts.index, autopct='%1.1f%%',
       colors=COLORS, startangle=140, pctdistance=0.82,
       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax.set_title('Payment Method Distribution')

# ── Chart 5 : Order Status (donut) ────────────────────────────
ax = axes[4]
status_counts = df['Order_Status'].value_counts()
wedges, texts, autotexts = ax.pie(
    status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
    colors=['#55A868','#C44E52','#DD8452'],
    startangle=90, pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'width': 0.6})
ax.set_title('Order Status Breakdown')

# ── Chart 6 : Category Orders Count (bar) ─────────────────────
ax = axes[5]
ax.bar(cat_ord.index, cat_ord.values, color=COLORS, edgecolor='white')
ax.set_title('Number of Orders by Category')
ax.set_xlabel('Category')
ax.set_ylabel('Orders')
ax.tick_params(axis='x', rotation=30)
for i, v in enumerate(cat_ord.values):
    ax.text(i, v + 1, str(v), ha='center', fontsize=8)

# ── Chart 7 : Customer Rating Distribution (hist) ─────────────
ax = axes[6]
ax.hist(df['Customer_Rating'], bins=15, color='#8172B2', edgecolor='white', linewidth=0.5)
ax.set_title('Customer Rating Distribution')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
avg_rating = df['Customer_Rating'].mean()
ax.axvline(avg_rating, color='red', linestyle='--', linewidth=1.5,
           label=f'Avg: {avg_rating:.2f}')
ax.legend()

# ── Chart 8 : Quarterly Revenue (bar) ────────────────────────
ax = axes[7]
q_rev = df.groupby('Quarter')['Revenue'].sum().sort_index()
bars = ax.bar(q_rev.index, q_rev.values, color=COLORS[:4], edgecolor='white')
ax.set_title('Quarterly Revenue Comparison')
ax.set_xlabel('Quarter')
ax.set_ylabel('Revenue (₹)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
            f'₹{bar.get_height()/1e6:.1f}M', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('ecommerce_dashboard.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Dashboard saved → ecommerce_dashboard.png")

# ════════════════════════════════════════════════════════════
# STEP 5 — KEY INSIGHTS REPORT
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 5 : KEY BUSINESS INSIGHTS")
print("="*55)

top2_cat   = cat_rev.index[:2].tolist()
low_cat    = cat_rev.index[-1]
ret_rate   = (df['Order_Status'] == 'Returned').mean() * 100
can_rate   = (df['Order_Status'] == 'Cancelled').mean() * 100
upi_pct    = (df['Payment_Method'] == 'UPI').mean() * 100
best_q     = q_rev.idxmax()

insights = f"""
  1. REVENUE   → Total revenue ₹{total_revenue:,.0f} from {total_orders:,} orders.
  2. TOP CATS  → {top2_cat[0]} & {top2_cat[1]} drive highest revenue.
  3. LOW CAT   → {low_cat} has lowest revenue — needs promotion.
  4. CITY      → {city_rev.index[0]} is the top performing city.
  5. MONTH     → {best_month} recorded peak sales.
  6. QUARTER   → {best_q} was the strongest quarter.
  7. PAYMENT   → UPI accounts for {upi_pct:.1f}% of transactions.
  8. RETURNS   → Return rate {ret_rate:.1f}% | Cancel rate {can_rate:.1f}%.
  9. RATINGS   → Average customer rating {avg_rating:.2f} / 5.0.
  10. INSIGHT  → Focus marketing budget on {city_rev.index[0]} & {city_rev.index[1]}.
"""
print(insights)

# Save insights to text file
with open('key_insights.txt', 'w', encoding='utf-8') as f:
    f.write("E-COMMERCE SALES ANALYSIS — KEY INSIGHTS\n")
    f.write("Analyst: Sakthiganesh K\n")
    f.write("="*50 + "\n")
    f.write(insights)

# ════════════════════════════════════════════════════════════
# STEP 6 — EXPORT SUMMARY TO EXCEL
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 6 : EXPORTING SUMMARY TO EXCEL")
print("="*55)

with pd.ExcelWriter('ecommerce_summary.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    cat_rev.reset_index().rename(columns={'Revenue':'Total Revenue'}).to_excel(
        writer, sheet_name='Category Revenue', index=False)
    monthly_rev.reset_index().rename(columns={'Revenue':'Monthly Revenue'}).to_excel(
        writer, sheet_name='Monthly Revenue', index=False)
    city_rev.reset_index().rename(columns={'Revenue':'City Revenue'}).to_excel(
        writer, sheet_name='City Revenue', index=False)
    pay_counts.reset_index().rename(columns={'count':'Orders'}).to_excel(
        writer, sheet_name='Payment Methods', index=False)

print("  ✅ Excel exported → ecommerce_summary.xlsx")

print("\n" + "="*55)
print("  ✅ PROJECT COMPLETE!")
print("="*55)
print("""
  Files created:
  📄 ecommerce_sales_data.csv   — raw dataset (1000 rows)
  📊 ecommerce_dashboard.png    — 8-chart visual dashboard
  📝 key_insights.txt           — business insights report
  📗 ecommerce_summary.xlsx     — Excel summary workbook
  🐍 ecommerce_analysis.py      — full Python analysis code
""")
