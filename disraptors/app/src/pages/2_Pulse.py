import time
import re
import pandas as pd
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

pages_folder = Path(__file__).parent
app_folder = pages_folder.parent.parent

# Creating the Layout of the page
st.set_page_config(
    page_title="Pulse",
    layout="wide"
)
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    h1 {padding: 0px; }
</style>
""", unsafe_allow_html=True)

st.title(":chart_with_upwards_trend: Pulse check on Vanguard fund.")
st.caption("🚀 Powered by Claude, built on reddit, twitter and you-tube interactions.")
"---"

# @st.cache_data
def load_data(df_type):
    data_path = app_folder / "output_datasets"
    if df_type == "pulse":
        return pd.read_excel(data_path / "pulse_output.xlsx")
    elif df_type == "cashflow":
        return pd.read_csv(data_path / "etf-cashflow-summary-clean.csv")
    elif df_type == "similar":
        return pd.read_parquet(data_path / "canada_sim_scores")
    elif df_type == "mapping":
        return pd.read_csv(data_path / "us_mf_etf_fi_pf_stats_end 3.csv", dtype=str)

df_pulse = load_data("pulse")
cashflow_df = load_data("cashflow")
cashflow_df.loc[:, '19-Jan':'24-Feb'].replace(",", "", regex=True, inplace=True)
similarity_df = load_data("similar")
similarity_df.drop(columns=similarity_df.columns[0], axis=1, inplace=True)
mapping_df = load_data("mapping")


selected_fund = st.selectbox(label = "_Select the Vanguard fund._", options = df_pulse["ticker"])
def typewriter(text: str, speed: int):
    tokens = re.findall(r"\S+|\n", text)
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

with st.expander(":blue[Summarized social media sentiment.]"):
    with st.spinner('Wait for it...'):
        time.sleep(0.5)
    summary = df_pulse.query(f"ticker == '{selected_fund}'")["synopsis"].iloc[0]
    # st.markdown(df_pulse.query(f"ticker == '{selected_fund}'")["synopsis"].iloc[0])
    typewriter(text=summary, speed=150)

# @st.cache_data
def _find_top_similar_tickers(target_ticker):    
    target_secid = mapping_df.loc[mapping_df['Ticker']==target_ticker]['SecId'].values[0]
    similar_secids = similarity_df.loc[similarity_df['SecId']==target_secid][['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']].values[0]
    similar_scores = []
    similar_secids_ = []
    similars = {}
    for item in similar_secids:
        similar_secids_.append(item.split(':')[0])
        similar_scores.append(item.split(':')[1])
        similars.update({item.split(':')[0]: item.split(':')[1]})
    
    similar_tickers = []
    for sec_id in similar_secids_:
        try:
            similar_tickers.append(mapping_df.loc[mapping_df['SecId']==sec_id]['Ticker'].values[0])
            similars[mapping_df.loc[mapping_df['SecId']==sec_id]['Ticker'].values[0]] = similars[sec_id]
            del similars[sec_id]
        except Exception:
            continue  
    
    return similar_tickers, similar_scores, similars

# @st.cache_data
def _plot_cashflows(ticker, history_window_length, top_k, include_one_interal_vanguard):
    similar_tickers, similar_scores, similars = _find_top_similar_tickers(ticker)
    # flitering tickers from secid
    similars = {key: similars[key] for key in similars if len(key)<=5}
    # check for vangaurd internal, if needed
    for v_rank, tck in enumerate(similars.keys()):
        if tck.startswith('V'):
            v_ticker = tck
            
    if include_one_interal_vanguard and v_rank > top_k:
        tups = list(similars.items())
        tups[top_k-1], tups[v_rank-1] = tups[v_rank-1], tups[top_k-1]
        similars = dict(tups)
    # select top-k based on user perefence
    top_similar_tickers = list(similars.keys())[:top_k]

    cash_flow = cashflow_df.loc[cashflow_df['ETF']==ticker]
    similar_cash_flows = cashflow_df.loc[cashflow_df['ETF'].isin(top_similar_tickers)]

    months = cashflow_df.columns.to_list()
    months.remove('ETF')
    months.remove('Fund Name')


    cash_flow_values = cash_flow[months].values[0]
    cash_flow_values = np.asarray(cash_flow_values, dtype=float)

    similar_cash_flow_values = similar_cash_flows[months].values
    similar_cash_flow_values = np.asarray(similar_cash_flow_values, dtype=float)

    x = np.arange(1,len(months)+1)
    y = cash_flow_values
    ys = similar_cash_flow_values
    x_ticks_labels = months

    
    df = pd.DataFrame(dict(
        month = list(x_ticks_labels[-history_window_length:]),
        cashflow = list(y[-history_window_length:]),
        ticker = [f'{cash_flow["ETF"].values[0]}'] * history_window_length))

    for i in range(min(top_k, len(similar_cash_flows))):
        temp_df = pd.DataFrame(dict(
            month = list(x_ticks_labels[-history_window_length:]),
            cashflow = list(ys[i][-history_window_length:]),
            ticker = [f'{similar_cash_flows["ETF"].values[i]} (similarity score: {similars[similar_cash_flows["ETF"].values[i]]})'] * history_window_length
        ))
        df = pd.concat([df, temp_df], ignore_index=True, sort=False)


    fig = px.line(df, x='month', y='cashflow', color='ticker', symbol="ticker")
    return fig

# @st.cache_data
def find_similar_tickers_and_plot_cashflows(ticker, history_window_length=12, top_k = 3, include_one_interal_vanguard=False):
    global mapping_df
    mapping_df = mapping_df[['SecId', 'Ticker']]
    mapping_df = mapping_df.dropna().reset_index(drop=True)
    return _plot_cashflows(ticker, history_window_length, top_k, include_one_interal_vanguard)


with st.expander(":blue[Cash flow and similarity insights.]"):
    with st.spinner('Wait for it...'):
        time.sleep(0.5)
    fig = find_similar_tickers_and_plot_cashflows(ticker=selected_fund, history_window_length=50, top_k = 3)
    st.plotly_chart(fig, use_container_width=True)
