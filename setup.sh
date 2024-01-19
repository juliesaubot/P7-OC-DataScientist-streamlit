mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"juliesaubot@hotmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $8000\n\
host = 127.0.0.1\n\
" > ~/.streamlit/config.toml
