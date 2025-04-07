import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .main-header {font-size:36px; font-weight:bold; margin-bottom:20px;}
        .sub-header {font-size:24px; margin-top:20px; margin-bottom:10px;}
        .stTabs [data-baseweb="tab-list"] {gap: 24px;}
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #007af5 !important;
            color: white !important;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: bold;
        }
        div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] p {
            font-size: 16px;
        }
        .sentiment-positive {color: green; font-weight: bold;}
        .sentiment-negative {color: red; font-weight: bold;}
        .sentiment-neutral {color: #ff9d00; font-weight: bold;}
        .stProgress .st-bo {
            background-color: #1c83e1;
        }
        div.block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# Configure Gemini API
api_key = st.sidebar.text_input("Enter your Gemini API Key:", value="AIzaSyDmGOBxL9RZ31OCrNIUG4YhWfW_rJogWY0", type="password")
if api_key:
    genai.configure(api_key=api_key)
    
    # Load Gemini model (only when API key is provided)
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.sidebar.error(f"Error connecting to Gemini API: {str(e)}")
        st.stop()
else:
    st.sidebar.warning("Please enter your Gemini API key to continue")
    st.stop()

# Sidebar with app information
with st.sidebar:
    st.markdown("## About")
    st.markdown("This app analyzes sentiment using Google's Gemini AI model.")
    st.markdown("### Features:")
    st.markdown("- YouTube video analysis")
    st.markdown("- Direct text analysis")
    st.markdown("- Batch analysis with CSV files")
    
    st.markdown("### How sentiment is scored:")
    st.markdown("- **+1.0**: Very positive")
    st.markdown("- **0.0**: Neutral")
    st.markdown("- **-1.0**: Very negative")
    
    st.markdown("---")
    st.markdown("### CSV File Format")
    st.markdown("Your CSV file should contain at least one column with text to analyze. The column name should include one of these words: 'text', 'content', 'comment', or 'description'.")
    
    sample_data = {
        'text': ['I love this product!', 'This is terrible service', 'The item was okay.'],
    }
    sample_df = pd.DataFrame(sample_data)
    
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_sentiment_data.csv",
        mime="text/csv",
    )

# --- Helper Functions ---

# Extract text from YouTube video (transcript or description)
def extract_youtube_text(video_url):
    try:
        video_id = re.search(r"(?:v=|youtu\.be/)([^&#]+)", video_url).group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([line['text'] for line in transcript])
        return text, "transcript"
    except Exception:
        try:
            yt = YouTube(video_url)
            return yt.description or "No description available.", "description"
        except:
            return "Could not extract YouTube text.", "error"

# Analyze sentiment using Gemini
def analyze_sentiment(text):
    if not text or not text.strip():
        return "Empty text provided. Please enter content to analyze."
    
    prompt = f"""Analyze the sentiment of the following text. 
    First, provide a clear label of either **Positive**, **Negative**, or **Neutral**.
    Then, provide a short explanation of your analysis (2-3 sentences maximum).
    Finally, on a separate line, provide a sentiment score between -1.0 (very negative) and 1.0 (very positive) with 0.0 being neutral.
    Format the score as "Score: X.X" where X.X is a number between -1.0 and 1.0.
    
    Text to analyze: \"\"\" {text[:4000]} \"\"\" """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

# Parse sentiment score from Gemini response
def extract_sentiment_score(sentiment_result):
    try:
        # First try to find "Score: X.X" pattern
        match = re.search(r"Score:\s*(-?\d+\.\d+)", sentiment_result)
        if match:
            score = float(match.group(1))
        else:
            # Try finding any floating point number
            match = re.search(r"(-?\d+\.\d+)", sentiment_result)
            if match:
                score = float(match.group(1))
            else:
                # Fallback to keyword-based estimation
                text = sentiment_result.lower()
                if "very positive" in text:
                    score = 0.9
                elif "positive" in text:
                    score = 0.6
                elif "very negative" in text:
                    score = -0.9
                elif "negative" in text:
                    score = -0.6
                elif "neutral" in text:
                    score = 0.0
                else:
                    score = 0.0
        
        # Ensure score is within -1 to 1 range
        return max(-1.0, min(score, 1.0))
    except:
        # Default to neutral if parsing fails
        return 0.0

# Create a gauge chart for sentiment visualization
def create_sentiment_gauge(score):
    # Define colors based on score
    if score < -0.33:
        color = "red"
    elif score > 0.33:
        color = "green"
    else:
        color = "gold"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.33], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [-0.33, 0.33], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [0.33, 1], 'color': 'rgba(0, 255, 0, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

# Extract sentiment category (Positive, Negative, Neutral)
def extract_sentiment_category(sentiment_result):
    text = sentiment_result.lower()
    if "very positive" in text:
        return "Very Positive"
    elif "positive" in text:
        return "Positive"
    elif "very negative" in text:
        return "Very Negative"
    elif "negative" in text:
        return "Negative"
    else:
        return "Neutral"

# Function to format sentiment result with highlighting
def format_sentiment_result(result):
    category = extract_sentiment_category(result)
    if "Positive" in category:
        color_class = "sentiment-positive"
    elif "Negative" in category:
        color_class = "sentiment-negative"
    else:
        color_class = "sentiment-neutral"
    
    # Add colored category at the beginning
    highlighted = f"<span class='{color_class}'>{category}</span><br><br>{result}"
    return highlighted

# Process CSV file for batch sentiment analysis
def process_csv(file):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        df = pd.read_csv(file)
        
        # Ensure there's a text column
        text_columns = [col for col in df.columns if any(x in col.lower() for x in ['text', 'content', 'comment', 'description'])]
        
        if not text_columns:
            return None, "No text column found in CSV. Please ensure your CSV has a column with 'text', 'content', 'comment', or 'description' in its name."
        
        text_column = text_columns[0]
        st.info(f"Using column '{text_column}' for sentiment analysis.")
        
        # Sample only first 50 rows if file is large to avoid overloading
        original_row_count = len(df)
        if len(df) > 50:
            st.warning(f"⚠️ CSV contains {original_row_count} rows. Analyzing only the first 50 rows to avoid API limits.")
            df = df.head(50)
        
        # Add sentiment analysis columns
        results = []
        total_rows = len(df)
        
        for i, row in enumerate(df[text_column].astype(str)):
            status_text.text(f"Analyzing row {i+1} of {total_rows}...")
            progress_bar.progress((i+1)/total_rows)
            
            if not row or row.isspace():
                sentiment_result = "Empty text"
                sentiment_score = 0.0
                sentiment_category = "Neutral"
            else:
                sentiment_result = analyze_sentiment(row)
                sentiment_score = extract_sentiment_score(sentiment_result)
                sentiment_category = extract_sentiment_category(sentiment_result)
            
            results.append({
                'sentiment_analysis': sentiment_result,
                'sentiment_score': sentiment_score,
                'sentiment_category': sentiment_category
            })
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Join with original DataFrame
        df = pd.concat([df, results_df], axis=1)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return df, None
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        return None, f"Error processing CSV: {str(e)}"

# Main app
st.markdown('<h1 class="main-header">🔍 Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("Analyze sentiment from YouTube videos, text input, or CSV files .")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs([
    "📹 YouTube Analysis", 
    "✏️ Text Analysis", 
    "📊 Batch Analysis (CSV)"
])

# Tab 1: YouTube Analysis
with tab1:
    st.markdown('<h2 class="sub-header">YouTube Video Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        youtube_link = st.text_input("Enter YouTube Video Link:", 
                                     placeholder="https://www.youtube.com/watch?v=...",
                                     key="youtube_input")
    with col2:
        analyze_yt_button = st.button("Analyze YouTube Video", key="analyze_yt")
    
    if youtube_link and analyze_yt_button:
        if "youtube.com" in youtube_link or "youtu.be" in youtube_link:
            with st.spinner("📥 Extracting content from YouTube..."):
                text, source_type = extract_youtube_text(youtube_link)
                
                if text and len(text) > 10 and source_type != "error":
                    st.success(f"✅ Successfully extracted YouTube {source_type}!")
                    
                    st.markdown('<h3 class="sub-header">📄 Extracted YouTube Content:</h3>', unsafe_allow_html=True)
                    with st.expander("Show extracted content"):
                        st.write(text[:2000] + "..." if len(text) > 2000 else text)
                    
                    with st.spinner("🧠 Analyzing sentiment with Gemini..."):
                        sentiment_result = analyze_sentiment(text)
                        sentiment_score = extract_sentiment_score(sentiment_result)
                        sentiment_category = extract_sentiment_category(sentiment_result)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown('<h3 class="sub-header">🧠 Sentiment Analysis Result:</h3>', unsafe_allow_html=True)
                        st.markdown(format_sentiment_result(sentiment_result), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<h3 class="sub-header">📊 Sentiment Score:</h3>', unsafe_allow_html=True)
                        fig = create_sentiment_gauge(sentiment_score)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Couldn't extract enough content from the YouTube video.")
        else:
            st.error("⚠️ Please enter a valid YouTube link (must include youtube.com or youtu.be).")

# Tab 2: Text Analysis
with tab2:
    st.markdown('<h2 class="sub-header">Analyze Any Text</h2>', unsafe_allow_html=True)
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=200,
        placeholder="Type or paste your text here...",
        key="text_input"
    )
    
    analyze_text_button = st.button("Analyze Text Sentiment", type="primary", key="analyze_text")
    
    if analyze_text_button:
        if text_input and len(text_input.strip()) > 0:
            with st.spinner("🧠 Analyzing sentiment with Gemini..."):
                sentiment_result = analyze_sentiment(text_input)
                sentiment_score = extract_sentiment_score(sentiment_result)
                sentiment_category = extract_sentiment_category(sentiment_result)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<h3 class="sub-header">🧠 Sentiment Analysis Result:</h3>', unsafe_allow_html=True)
                st.markdown(format_sentiment_result(sentiment_result), unsafe_allow_html=True)
            
            with col2:
                st.markdown('<h3 class="sub-header">📊 Sentiment Score:</h3>', unsafe_allow_html=True)
                fig = create_sentiment_gauge(sentiment_score)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ Please enter some text to analyze.")

# Tab 3: CSV Batch Analysis
with tab3:
    st.markdown('<h2 class="sub-header">Batch Analysis with CSV</h2>', unsafe_allow_html=True)
    
    upload_col, sample_col = st.columns([2, 1])
    
    with upload_col:
        st.markdown("Upload a CSV file with text to analyze multiple entries at once.")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="The CSV should have a column with 'text', 'content', 'comment', or 'description' in its name.",
            key="csv_uploader"
        )
    
    process_csv_button = st.button("Process CSV File", type="primary", key="process_csv", disabled=uploaded_file is None)
    
    if uploaded_file is not None and process_csv_button:
        st.markdown('<h3 class="sub-header">Processing CSV...</h3>', unsafe_allow_html=True)
        with st.spinner("🔄 Analyzing sentiments for each row..."):
            results_df, error_message = process_csv(uploaded_file)
            
            if error_message:
                st.error(f"❌ {error_message}")
            elif results_df is not None:
                st.success(f"✅ Successfully analyzed {len(results_df)} entries!")
                
                # Display results in tabs
                result_tab1, result_tab2 = st.tabs(["📊 Analysis Results", "📈 Visualizations"])
                
                with result_tab1:
                    st.dataframe(
                        results_df,
                        column_config={
                            "sentiment_score": st.column_config.NumberColumn(
                                "Sentiment Score",
                                format="%.2f",
                                help="Score from -1.0 (negative) to 1.0 (positive)"
                            ),
                            "sentiment_category": st.column_config.TextColumn(
                                "Category",
                                help="Sentiment classification"
                            ),
                            "sentiment_analysis": st.column_config.TextColumn(
                                "Analysis",
                                width="large",
                                help="Full sentiment analysis text"
                            )
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add download button for analyzed results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download analyzed results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )
                
                with result_tab2:
                    # Create visualizations if we have results
                    if not results_df.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sentiment distribution pie chart
                            sentiment_counts = results_df['sentiment_category'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            
                            fig1 = px.pie(
                                sentiment_counts, 
                                values='Count', 
                                names='Sentiment', 
                                title='Sentiment Distribution',
                                color='Sentiment',
                                color_discrete_map={
                                    'Very Positive': 'darkgreen',
                                    'Positive': 'lightgreen',
                                    'Neutral': 'gold',
                                    'Negative': 'tomato',
                                    'Very Negative': 'darkred'
                                },
                                hole=0.4
                            )
                            fig1.update_traces(textposition='inside', textinfo='percent+label')
                            fig1.update_layout(
                                legend_title_text='Sentiment',
                                title={
                                    'text': 'Sentiment Distribution',
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 20}
                                }
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Sentiment scores histogram
                            fig2 = px.histogram(
                                results_df, 
                                x='sentiment_score',
                                nbins=10,
                                title='Sentiment Score Distribution',
                                color_discrete_sequence=['skyblue'],
                                histnorm='probability density'
                            )
                            fig2.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")
                            fig2.update_layout(
                                xaxis_title="Sentiment Score", 
                                yaxis_title="Frequency",
                                title={
                                    'text': 'Sentiment Score Distribution',
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 20}
                                }
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                            
                        # Add additional visualization - average sentiment by category if we have category columns
                        category_cols = [col for col in results_df.columns if any(x in col.lower() for x in ['category', 'group', 'type', 'class'])]
                        if category_cols:
                            category_col = category_cols[0]
                            st.markdown(f"#### Average Sentiment by {category_col}")
                            
                            # Calculate average sentiment by category
                            cat_sentiment = results_df.groupby(category_col)['sentiment_score'].mean().reset_index()
                            cat_sentiment = cat_sentiment.sort_values('sentiment_score')
                            
                            # Create horizontal bar chart
                            fig3 = px.bar(
                                cat_sentiment,
                                x='sentiment_score',
                                y=category_col,
                                orientation='h',
                                color='sentiment_score',
                                color_continuous_scale='RdYlGn',
                                title=f'Average Sentiment by {category_col}'
                            )
                            fig3.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")
                            fig3.update_layout(
                                xaxis_title="Average Sentiment Score",
                                yaxis_title=category_col,
                                height=max(300, len(cat_sentiment) * 30)
                            )
                            st.plotly_chart(fig3, use_container_width=True)


