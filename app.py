"""
Streamlit app for BERT-based news topic classifier.
This app provides an interactive interface for classifying news headlines.
"""

import streamlit as st
import pandas as pd
import time
import os
from model_utils import NewsClassifier, preprocess_text, get_sample_headlines

# Try to import plotly, fallback to simple charts if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Page configuration
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .sample-text {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Load the classifier model (cached for performance)"""
    classifier = NewsClassifier()
    success = classifier.load_model()
    return classifier, success

def create_confidence_chart(probabilities):
    """Create a confidence chart for prediction probabilities"""
    if not probabilities:
        return None
    
    if not PLOTLY_AVAILABLE:
        # Fallback to simple text display
        return None
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='Blues',
                showscale=False
            ),
            text=[f'{prob:.3f}' for prob in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="News Category",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def display_model_info(classifier):
    """Display model information in the sidebar"""
    st.sidebar.header("🤖 Model Information")
    
    info = classifier.get_model_info()
    
    if "error" in info:
        st.sidebar.error(info["error"])
        return
    
    st.sidebar.write(f"**Base Model:** {info.get('base_model', 'Unknown')}")
    st.sidebar.write(f"**Classes:** {info.get('num_classes', 'Unknown')}")
    st.sidebar.write(f"**Max Length:** {info.get('max_length', 'Unknown')} tokens")
    st.sidebar.write(f"**Device:** {info.get('device', 'Unknown')}")
    
    # Performance metrics if available
    if 'test_accuracy' in info:
        st.sidebar.metric("Test Accuracy", f"{info['test_accuracy']:.3f}")
    if 'test_f1_weighted' in info:
        st.sidebar.metric("F1-Score (Weighted)", f"{info['test_f1_weighted']:.3f}")
    
    # Class names
    st.sidebar.write("**Categories:**")
    for i, class_name in enumerate(info.get('class_names', [])):
        st.sidebar.write(f"  {i}: {class_name}")

def main():
    # Header
    st.markdown('<h1 class="main-header">📰 News Topic Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Classify news headlines into categories using a fine-tuned BERT model**")
    
    # Load model
    with st.spinner("Loading BERT model..."):
        classifier, model_loaded = load_classifier()
    
    if not model_loaded:
        st.warning("⚠️ **Model not ready yet**")
        st.info("This is a BERT-based news topic classifier. To get started:")
        st.info("1. Install required packages: `pip install torch transformers datasets scikit-learn plotly pandas joblib`")
        st.info("2. Run the `news_classifier_training.ipynb` notebook to train the model")
        st.info("3. Come back here to test your trained classifier!")
        
        # Show demo mode
        st.markdown("---")
        st.header("🎯 Demo Mode")
        st.write("Try out the interface with sample classifications:")
        
        demo_text = st.text_area(
            "Enter a news headline:",
            placeholder="Apple reports record quarterly earnings beating analyst expectations",
            height=100
        )
        
        if st.button("🎲 Demo Classification", type="primary"):
            if demo_text.strip():
                # Simple demo logic
                demo_text_lower = demo_text.lower()
                if any(word in demo_text_lower for word in ['sports', 'game', 'team', 'player', 'match', 'football', 'basketball']):
                    demo_result = "Sports"
                elif any(word in demo_text_lower for word in ['business', 'earnings', 'company', 'stock', 'market', 'economy']):
                    demo_result = "Business"
                elif any(word in demo_text_lower for word in ['science', 'technology', 'ai', 'research', 'discovery', 'tech']):
                    demo_result = "Science/Technology"
                else:
                    demo_result = "World"
                
                st.markdown(
                    f'<div class="prediction-result">'
                    f'<h3>🎯 Demo Prediction</h3>'
                    f'<p><strong>Category:</strong> {demo_result}</p>'
                    f'<p><strong>Note:</strong> This is a simple demo. Train the model for real AI-powered classification!</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        return
    
    st.success("✅ **Model loaded successfully!**")
    
    # Display model info in sidebar
    display_model_info(classifier)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Options")
    show_probabilities = st.sidebar.checkbox("Show all class probabilities", value=True)
    show_examples = st.sidebar.checkbox("Show sample headlines", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Classify Your News Headline")
        
        # Text input
        user_input = st.text_area(
            "Enter a news headline to classify:",
            height=100,
            placeholder="Example: Apple reports record quarterly earnings beating analyst expectations",
            value=st.session_state.get("user_input", "")
        )
        
        # Predict button
        if st.button("🚀 Classify Headline", type="primary"):
            if user_input.strip():
                # Preprocess text
                processed_text = preprocess_text(user_input)
                
                # Show processing
                with st.spinner("Classifying..."):
                    # Make prediction
                    result = classifier.predict(processed_text, return_probabilities=show_probabilities)
                
                # Display results
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Main prediction result
                    st.markdown(
                        f'<div class="prediction-result">'
                        f'<h3>📊 Prediction Result</h3>'
                        f'<p><strong>Category:</strong> {result["predicted_class"]}</p>'
                        f'<p><strong>Confidence:</strong> {result["confidence"]:.3f} ({result["confidence"]*100:.1f}%)</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence visualization
                    if show_probabilities and 'all_probabilities' in result:
                        fig = create_confidence_chart(result['all_probabilities'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        elif not PLOTLY_AVAILABLE:
                            # Fallback to text display
                            st.write("**All Class Probabilities:**")
                            prob_data = result['all_probabilities']
                            for category, prob in sorted(prob_data.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"- **{category}**: {prob:.3f} ({prob*100:.1f}%)")
                            
                            # Create a simple bar chart using streamlit
                            chart_data = pd.DataFrame({
                                'Category': list(prob_data.keys()),
                                'Probability': list(prob_data.values())
                            })
                            st.bar_chart(chart_data.set_index('Category'))
            else:
                st.warning("⚠️ Please enter a news headline to classify.")
    
    with col2:
        st.header("📈 Quick Stats")
        
        # Model performance metrics
        info = classifier.get_model_info()
        if 'test_accuracy' in info:
            st.metric("Model Accuracy", f"{info['test_accuracy']:.1%}")
        if 'test_f1_weighted' in info:
            st.metric("F1-Score", f"{info['test_f1_weighted']:.3f}")
        
        st.metric("Classes", len(info.get('class_names', [])))
        
        # Category distribution (example)
        categories = info.get('class_names', [])
        if categories:
            st.subheader("📋 Categories")
            for category in categories:
                st.write(f"• {category}")
    
    # Sample headlines section
    if show_examples:
        st.header("💡 Try These Sample Headlines")
        
        sample_headlines = get_sample_headlines()
        
        # Create tabs for different categories
        tabs = st.tabs(["All Samples", "Business", "Sports", "World", "Science/Tech"])
        
        with tabs[0]:
            st.write("Click any headline below to test it:")
            
            for i, sample in enumerate(sample_headlines):
                col_text, col_button = st.columns([4, 1])
                
                with col_text:
                    st.markdown(
                        f'<div class="sample-text">'
                        f'<strong>Expected:</strong> {sample["expected"]}<br>'
                        f'<em>{sample["text"]}</em>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col_button:
                    if st.button(f"Test", key=f"sample_{i}"):
                        # Auto-fill the text area
                        st.session_state.auto_text = sample["text"]
                        st.rerun()
        
        # Category-specific tabs
        categories = ["Business", "Sports", "World", "Science/Technology"]
        for i, category in enumerate(categories, 1):
            with tabs[i]:
                category_samples = [s for s in sample_headlines if s["expected"] == category]
                if category_samples:
                    for j, sample in enumerate(category_samples):
                        if st.button(f"Test: {sample['text'][:50]}...", key=f"cat_{i}_{j}"):
                            st.session_state.auto_text = sample["text"]
                            st.rerun()
    
    # Handle auto-filled text
    if hasattr(st.session_state, 'auto_text'):
        st.session_state.user_input = st.session_state.auto_text
        del st.session_state.auto_text
        st.rerun()
    
    # Batch processing section
    st.header("📦 Batch Classification")
    
    with st.expander("Upload multiple headlines for batch processing"):
        uploaded_file = st.file_uploader(
            "Upload a CSV file with headlines",
            type=['csv'],
            help="CSV should have a column named 'text' containing the headlines"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column with headlines")
                else:
                    st.write(f"Loaded {len(df)} headlines")
                    
                    if st.button("🚀 Classify All"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, text in enumerate(df['text']):
                            result = classifier.predict(str(text), return_probabilities=False)
                            results.append({
                                'text': text,
                                'predicted_class': result.get('predicted_class', 'Error'),
                                'confidence': result.get('confidence', 0.0)
                            })
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Display results
                        results_df = pd.DataFrame(results)
                        st.write("### Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "classification_results.csv",
                            "text/csv"
                        )
                        
                        # Summary statistics
                        st.write("### Summary")
                        class_counts = results_df['predicted_class'].value_counts()
                        
                        if PLOTLY_AVAILABLE:
                            fig = px.bar(
                                x=class_counts.values,
                                y=class_counts.index,
                                orientation='h',
                                title="Classification Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback to simple text display
                            st.write("**Classification Results:**")
                            for category, count in class_counts.items():
                                st.write(f"- {category}: {count} headlines")
                            st.bar_chart(class_counts)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**About:** This classifier uses a fine-tuned BERT model trained on the AG News dataset. "
        "It can classify news headlines into four categories: World, Sports, Business, and Science/Technology."
    )
    
    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        st.write("**Model Architecture:** BERT (bert-base-uncased)")
        st.write("**Training Dataset:** AG News Dataset")
        st.write("**Framework:** Hugging Face Transformers + PyTorch")
        st.write("**Deployment:** Streamlit")
        
        info = classifier.get_model_info()
        if info:
            st.json(info)

if __name__ == "__main__":
    main()
