import streamlit as st
import requests

st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #3B9AE1;
            text-align: center;
        }
        .subheader {
            font-size: 30px;
            font-weight: 600;
            color: #3B9AE1;
            padding-top: 10px;
        }
        .input-box {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .result-box {
            border-radius: 10px;
            background-color: #F0F8FF;
            color:black;
            padding: 15px;
            margin-top: 20px;
        }
        .button {
            background-color: #3B9AE1;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
        .warning {
            color: #FF6347;
            font-size: 16px;
        }
        .error {
            color: #FF0000;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Byte Pair Encoding (BPE) with Streamlit</div>', unsafe_allow_html=True)
text = st.text_area("Enter Text for BPE Processing:", height=150, key="text_input")
num_merges = st.text_input("Enter the number of merges:", value="10", key="merges_input")
try:
    num_merges = int(num_merges) 
    if num_merges < 1:
        st.markdown('<div class="warning">Please enter a positive integer greater than 0.</div>', unsafe_allow_html=True)
except ValueError:
    num_merges = 10 
    st.markdown('<div class="warning">Invalid input! Using the default value of 10 merges.</div>', unsafe_allow_html=True)

if st.button("Process Text", key="process_button"):
    if text:
        response = requests.post("http://127.0.0.1:5000/bpe", json={"text": text, "num_merges": num_merges})
        if response.status_code == 200:
            result = response.json()
            encoded_tokens = result['encoded_tokens']
            decoded_text = result['decoded_text']
            vocab_size = result['vocab_size']
            num_merges_done = result['num_merges']

            st.markdown(f'<div class="subheader">Encoded Tokens</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{encoded_tokens}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="subheader">Decoded Text</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{decoded_text}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="subheader">Vocabulary Size</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{vocab_size}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="subheader">Number of Merges</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{num_merges_done}</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="error">Error in processing the text!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning">Please enter some text.</div>', unsafe_allow_html=True)
