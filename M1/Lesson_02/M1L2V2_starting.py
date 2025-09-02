from dotenv import load_dotenv
import openai
import streamlit as st

load_dotenv()

client = openai.OpenAI()

st.title("Hello, GenAI!")
st.write("This is my first Streamlit app.! (well it's not I was among the first adopters but it's been years so let's just pretend, ok?)")

response = client.responses.create(
    model="gpt-4o",
    input=[{"role": "user", "content": "Explain generative AI in one sentence."}
           ],
    temperature=0.7,
    max_output_tokens=100
)

st.write(response.output[0].content[0].text)