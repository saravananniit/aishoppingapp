import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.serper import SerperTools
from dotenv import load_dotenv
import pandas as pd
import re
from docx import Document
from io import BytesIO

# Load environment variables (SERPER_API_KEY required)
load_dotenv()

@st.cache_resource
def setup_agent():
    return Agent(
        name="shopping partner",
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            "You are a product recommender agent specializing in finding products that match user preferences.",
            "Prioritize finding products that satisfy as many user requirements as possible, but ensure a minimum match of 50%.",
            "Search for products only from authentic and trusted e-commerce websites such as Amazon, Flipkart, Myntra, Meesho, Google Shopping, Nike, and other reputable platforms.",
            "Verify that each product recommendation is in stock and available for purchase.",
            "Avoid suggesting counterfeit or unverified products.",
            "Clearly mention the key attributes of each product (e.g., price, brand, features) in the response.",
            "Format the recommendations neatly and ensure clarity for ease of user understanding.",
        ],
        tools=[SerperTools()],
        show_tool_calls=True,
    )

# Function to extract structured info from response
def parse_response_to_table(response_text: str):
    pattern = r"\d+\.\s(.+?)\n\s*- Price: ₹([\d,\.]+)\n\s*- Fabric: (.+?)\n\s*- Features: (.+?)\n\s*- Link: \[View on (.+?)\]\((https?://[^\)]+)\)"
    matches = re.findall(pattern, response_text)

    data = []
    for match in matches:
        title, price, fabric, features, site, link = match
        data.append({
            "Product": title.strip(),
            "Price (₹)": price.strip(),
            "Fabric": fabric.strip(),
            "Features": features.strip(),
            "Store": site.strip(),
            "Link": link.strip()
        })

    return pd.DataFrame(data)

# DOCX export
def generate_docx_from_df(df):
    doc = Document()
    doc.add_heading('Product Recommendations', 0)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Table Grid'

    hdr_cells = table.rows[0].cells
    for i, column in enumerate(df.columns):
        hdr_cells[i].text = column

    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, cell in enumerate(row):
            cells[i].text = str(cell)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- UI ---
st.set_page_config(page_title="🛍️ AI Shopping Assistant", layout="centered")
st.title("🛍️ AI Shopping Assistant")
st.markdown("Find the best products that match your preferences using AI and real-time web search.")

with st.form("product_form"):
    col1, col2 = st.columns(2)
    with col1:
        category = st.text_input("Product Category", "Sports shoe")
        color = st.text_input("Preferred Color", "Blue")
    with col2:
        purpose = st.text_input("Purpose", "Comfortable for long-distance running")
        budget = st.text_input("Max Budget (INR)", "10000")

    submitted = st.form_submit_button("🔍 Get Recommendations")

if submitted:
    with st.spinner("Searching and analyzing..."):
        agent = setup_agent()
        query = (
            f"I am looking for {category} with the following preferences: "
            f"Color: {color}, Purpose: {purpose}, Budget: Under Rs. {budget}"
        )
        result = agent.run(query)
        raw_text = result.content if hasattr(result, 'content') else str(result)

        # Clean markdown-style bold text
        cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_text)

        # Try structured parsing
        df = parse_response_to_table(cleaned_text)

    st.subheader("📋 Recommended Products")
    if not df.empty:
        st.dataframe(df, use_container_width=True)

        docx_file = generate_docx_from_df(df)
        st.download_button(
            label="📥 Download as Word Document",
            data=docx_file,
            file_name="product_recommendations.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    else:
        if any(keyword in cleaned_text.lower() for keyword in ["here are some", "options", "suitable"]):
            st.markdown(cleaned_text)
        else:
            st.info("No product recommendations were found.")
