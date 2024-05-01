import streamlit as st
from pathlib import Path

src_folder = Path(__file__).parent
app_folder = src_folder.parent.absolute()

st.header(":people_holding_hands: DisRaptors Team", divider='rainbow')
st.markdown("**Shivendra Pandey (Shiv)**")
st.markdown("Principal Machine Learning Engineer")
st.text("")
st.markdown("**Hong Huang**")
st.markdown("Senior Machine Learning Scientist")
st.text("")
st.markdown("**Erfan Pirmorad**")
st.markdown("Data Scientist")
st.text("")
st.markdown("**Ali El-Khatib**")
st.markdown("Machine Learning Engineer")
st.text("")
st.markdown("**Ashish Dewan**")
st.markdown("Investment Consultant")

st.text("")
st.header(":female-detective: Business Objective", divider='rainbow')
st.subheader(":blue[Problem]")
st.subheader(":blue[Proposed Solution]")
st.subheader(":blue[Product Offering]")

st.text("")
st.header(":twisted_rightwards_arrows: Architecture Flow", divider='rainbow')
st.image(f"{app_folder}/images/architecture.gif", use_column_width="always")