from turtle import title
import streamlit as st
import helper as help
import altair as alt

st.set_page_config(
    page_title="Bank Portugis",
    page_icon="👋",
)


dt = st.radio(
        "Select Dataset",
        ('Train', 'Test'))

tab1, tab2 = st.tabs(["Categorical Data", "Distribution Data"])
with tab1:    
    selectbox = st.selectbox(
        "Select variable input",
        ('Job', 'Marital', 'Education', 'Defaut', 'Housing', 'Loan', 'Contact', 'Month', 'Poutcome'))

    if dt =="Train":
        chart = alt.Chart(help.train(), title="Count of "+selectbox).mark_bar().encode(
            x=alt.X(selectbox.lower(), axis=alt.Axis(labelAngle=45)),
            y=alt.Y('count()', title=None),
            color='y',
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        chart = alt.Chart(help.test(), title="Count of "+selectbox).mark_bar().encode(
            x=alt.X(selectbox.lower(), axis=alt.Axis(labelAngle=45)),
            y=alt.Y('count()', title=None),
            color='y',
        )
        st.altair_chart(chart, use_container_width=True)

with tab2:
    selectbox_num = st.selectbox(
        "Select variable input",
        ('Age', 'Balance', 'Day', 'Duration', 'Campaign', 'Pdays', 'Previous'))

    if dt =="Train":
        bp = alt.Chart(help.train()).mark_boxplot().encode(
        x=selectbox_num.lower(),
        y='y'
        )
        st.altair_chart(bp, use_container_width=True)
    else:
        bp = alt.Chart(help.test()).mark_boxplot().encode(
        x=selectbox_num.lower(),
        y='y'
        )
        st.altair_chart(bp, use_container_width=True)
