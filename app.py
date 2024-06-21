import streamlit as st
from dotenv import load_dotenv
from utils import *


def main():
    load_dotenv()

    st.set_page_config(page_title="Extracto-v0.1", layout="wide")
    
    st.markdown(
        """
        <style>
            .header {
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                padding: 20px;
            }
            .subheader {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                padding-bottom: 20px;
            }
            .file-uploader {
                padding: 20px;
            }
            .extract-button {
                text-align: center;
                padding-top: 20px;
            }
            .spinner-text {
                text-align: center;
                padding-top: 20px;
                font-size: 18px;
            }
            .success-message {
                text-align: center;
                padding-top: 20px;
                font-size: 24px;
                color: green;
            }
        </style>
        """
    , unsafe_allow_html=True)

    st.markdown("<h1 class='header'>Extracto-v0.1</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>I can help you in extracting invoice data</h2>", unsafe_allow_html=True)
    close_server = st.button("close server")
    if close_server:
        parent_pid = os.getpid()
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()
        parent.kill()

    # Upload the Invoices (pdf files)
    pdf = st.file_uploader("Upload invoices here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Extract Data")

    if submit:
        with st.spinner('Wait for it...'):
            df = create_docs(pdf)
            st.write(df.head())

            data_as_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download data as CSV",
                data_as_csv,
                "benchmark-tools.csv",
                "text/csv",
                key="download-tools-csv",
            )
        st.markdown("<p class='success-message'>Hope I was able to save your time❤️</p>", unsafe_allow_html=True)


# Invoking main function
if __name__ == '__main__':
    main()
