import streamlit as st

st.set_page_config(
    page_title="Workload Profiler",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

home       = st.Page("pages/0_Home.py",               title="Home",                icon="🏠", default=True)
explorer   = st.Page("pages/1_Cluster_Explorer.py",   title="Cluster Explorer",    icon="🔍")
classifier = st.Page("pages/2_Workload_Classifier.py", title="Workload Classifier", icon="🎯")
findings   = st.Page("pages/3_Key_Findings.py",       title="Key Findings",        icon="📊")

pg = st.navigation({"": [home], "Explore": [explorer, classifier, findings]})
pg.run()
