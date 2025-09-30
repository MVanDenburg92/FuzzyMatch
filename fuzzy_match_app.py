"""
Fuzzy Matching Streamlit Application
A web-based interface for performing fuzzy matching between CSV files.

Installation:
pip install streamlit pandas rapidfuzz

Run with:
streamlit run fuzzy_match_app.py
"""

import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Tuple

# Try to import rapidfuzz first (faster), fall back to fuzzywuzzy
USING_RAPIDFUZZ = False
try:
    from rapidfuzz import fuzz, process
    USING_RAPIDFUZZ = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        USING_RAPIDFUZZ = False
    except ImportError:
        st.error(
            "Neither rapidfuzz nor fuzzywuzzy is installed. "
            "Please install one of them:\n"
            "```pip install rapidfuzz```  (recommended, faster)\n"
            "OR\n"
            "```pip install fuzzywuzzy python-Levenshtein```"
        )
        st.stop()


class FuzzyMatcher:
    """A class to perform fuzzy matching between multiple CSV files."""
    
    def __init__(self, threshold: int = 80):
        self.threshold = threshold
        self.dataframes = {}
    
    def _extract_match_info(self, match_result):
        """Helper function to extract match information from either rapidfuzz or fuzzywuzzy results."""
        if match_result is None:
            return None, 0, None
            
        if USING_RAPIDFUZZ:
            if len(match_result) == 3:
                return match_result[0], match_result[1], match_result[2]
            else:
                return match_result[0], match_result[1], None
        else:
            if len(match_result) == 3:
                return match_result[0], match_result[1], match_result[2]
            else:
                return match_result[0], match_result[1], None
    
    def load_dataframe(self, df: pd.DataFrame, name: str):
        """Load a DataFrame."""
        self.dataframes[name] = df
        return df
    
    def fuzzy_match_single_column(self, 
                                   df1_name: str, 
                                   df2_name: str,
                                   col1: str, 
                                   col2: str,
                                   method: str = 'token_sort_ratio') -> pd.DataFrame:
        """Perform fuzzy matching between two DataFrames on single columns."""
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        # Choose fuzzy matching method
        method_map = {
            'ratio': fuzz.ratio,
            'partial_ratio': fuzz.partial_ratio,
            'token_sort_ratio': fuzz.token_sort_ratio,
            'token_set_ratio': fuzz.token_set_ratio
        }
        
        fuzzy_func = method_map.get(method, fuzz.token_sort_ratio)
        
        results = []
        
        # Clean data
        df1[col1] = df1[col1].fillna('').astype(str)
        df2[col2] = df2[col2].fillna('').astype(str)
        
        choices = df2[col2].tolist()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df1)
        
        for idx, row in df1.iterrows():
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows}...")
            
            search_value = row[col1]
            
            if search_value.strip() == '':
                continue
            
            best_match = process.extractOne(search_value, choices, scorer=fuzzy_func)
            
            if best_match:
                match_value, score, match_idx = self._extract_match_info(best_match)
                
                if score >= self.threshold and match_idx is not None:
                    matched_row = df2.iloc[match_idx]
                    
                    result = {
                        f'{df1_name}_index': idx,
                        f'{df1_name}_{col1}': search_value,
                        f'{df2_name}_index': match_idx,
                        f'{df2_name}_{col2}': match_value,
                        'match_score': score
                    }
                    
                    for col in df1.columns:
                        if col != col1:
                            result[f'{df1_name}_{col}'] = row[col]
                    
                    for col in df2.columns:
                        if col != col2:
                            result[f'{df2_name}_{col}'] = matched_row[col]
                    
                    results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    
    def fuzzy_match_many_to_one(self,
                                 df1_name: str,
                                 df2_name: str,
                                 df1_columns: List[str],
                                 df2_column: str,
                                 combine_method: str = 'best') -> pd.DataFrame:
        """Match multiple columns from df1 against a single column in df2."""
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        results = []
        
        # Clean data
        for col in df1_columns:
            df1[col] = df1[col].fillna('').astype(str)
        df2[df2_column] = df2[df2_column].fillna('').astype(str)
        
        choices = df2[df2_column].tolist()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df1)
        
        for idx1, row1 in df1.iterrows():
            progress = (idx1 + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx1 + 1} of {total_rows}...")
            
            best_score = 0
            best_match_idx = None
            best_match_value = None
            matched_column = None
            column_scores = {}
            
            if combine_method == 'concatenate':
                combined_value = ' '.join([row1[col] for col in df1_columns if row1[col].strip() != ''])
                
                if combined_value.strip():
                    match_result = process.extractOne(combined_value, choices, scorer=fuzz.token_sort_ratio)
                    
                    if match_result:
                        best_match_value, best_score, best_match_idx = self._extract_match_info(match_result)
                        
                        if best_score >= self.threshold and best_match_idx is not None:
                            matched_column = 'combined'
                        
            else:  # 'best' method
                for col in df1_columns:
                    search_value = row1[col]
                    
                    if search_value.strip() == '':
                        column_scores[col] = 0
                        continue
                    
                    match_result = process.extractOne(search_value, choices, scorer=fuzz.token_sort_ratio)
                    
                    if match_result:
                        match_value, score, match_idx = self._extract_match_info(match_result)
                        column_scores[col] = score
                        
                        if score > best_score and match_idx is not None:
                            best_score = score
                            best_match_idx = match_idx
                            best_match_value = match_value
                            matched_column = col
            
            if best_score >= self.threshold and best_match_idx is not None:
                matched_row = df2.iloc[best_match_idx]
                
                result = {
                    f'{df1_name}_index': idx1,
                    f'{df2_name}_index': best_match_idx,
                    'match_score': best_score,
                    'matched_column': matched_column,
                    f'{df2_name}_{df2_column}': best_match_value
                }
                
                for col in df1_columns:
                    result[f'{df1_name}_{col}'] = row1[col]
                    if combine_method == 'best' and col in column_scores:
                        result[f'score_{col}'] = column_scores[col]
                
                for col in df1.columns:
                    if col not in df1_columns:
                        result[f'{df1_name}_{col}'] = row1[col]
                
                for col in df2.columns:
                    if col != df2_column:
                        result[f'{df2_name}_{col}'] = matched_row[col]
                
                results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)


# Streamlit App
def main():
    st.set_page_config(page_title="Fuzzy Matching Tool", page_icon="🔍", layout="wide")
    
    st.title("🔍 Fuzzy Matching Tool")
    st.markdown("Match records between CSV files using fuzzy string matching")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    threshold = st.sidebar.slider(
        "Match Threshold (%)",
        min_value=50,
        max_value=100,
        value=80,
        help="Minimum similarity score to consider a match (80 is recommended)"
    )
    
    matching_method = st.sidebar.selectbox(
        "Matching Algorithm",
        ["token_sort_ratio", "ratio", "partial_ratio", "token_set_ratio"],
        help="token_sort_ratio is usually best for names and addresses"
    )
    
    # Initialize session state
    if 'matcher' not in st.session_state:
        st.session_state.matcher = FuzzyMatcher(threshold=threshold)
    else:
        st.session_state.matcher.threshold = threshold
    
    # File upload section
    st.header("📁 Upload CSV Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset 1")
        file1 = st.file_uploader("Upload first CSV", type=['csv'], key='file1')
        
        if file1:
            df1 = pd.read_csv(file1)
            st.session_state.matcher.load_dataframe(df1, "Dataset_1")
            st.success(f"✅ Loaded: {len(df1)} rows, {len(df1.columns)} columns")
            
            with st.expander("Preview Dataset 1"):
                st.dataframe(df1.head())
    
    with col2:
        st.subheader("Dataset 2")
        file2 = st.file_uploader("Upload second CSV", type=['csv'], key='file2')
        
        if file2:
            df2 = pd.read_csv(file2)
            st.session_state.matcher.load_dataframe(df2, "Dataset_2")
            st.success(f"✅ Loaded: {len(df2)} rows, {len(df2.columns)} columns")
            
            with st.expander("Preview Dataset 2"):
                st.dataframe(df2.head())
    
    # Matching configuration
    if file1 and file2:
        st.header("🎯 Configure Matching")
        
        df1 = st.session_state.matcher.dataframes["Dataset_1"]
        df2 = st.session_state.matcher.dataframes["Dataset_2"]
        
        match_type = st.radio(
            "Select Matching Type",
            ["Single Column Match", "Multiple Columns to One"],
            horizontal=True
        )
        
        if match_type == "Single Column Match":
            col1, col2 = st.columns(2)
            
            with col1:
                df1_col = st.selectbox(
                    "Select column from Dataset 1",
                    options=df1.columns.tolist()
                )
            
            with col2:
                df2_col = st.selectbox(
                    "Select column from Dataset 2",
                    options=df2.columns.tolist()
                )
            
            if st.button("🚀 Run Matching", type="primary"):
                with st.spinner("Performing fuzzy matching..."):
                    results = st.session_state.matcher.fuzzy_match_single_column(
                        "Dataset_1",
                        "Dataset_2",
                        df1_col,
                        df2_col,
                        method=matching_method
                    )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} matches!")
                        
                        # Display statistics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Matches", len(results))
                        col2.metric("Average Score", f"{results['match_score'].mean():.1f}%")
                        col3.metric("Min Score", f"{results['match_score'].min():.1f}%")
                        
                        # Display results
                        st.subheader("📊 Matching Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv,
                            file_name="fuzzy_match_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No matches found above the threshold. Try lowering the threshold.")
        
        else:  # Multiple Columns to One
            st.markdown("**Select multiple columns from Dataset 1 to match against one column in Dataset 2**")
            
            df1_cols = st.multiselect(
                "Select columns from Dataset 1",
                options=df1.columns.tolist(),
                help="Select multiple columns that should be matched"
            )
            
            df2_col = st.selectbox(
                "Select single column from Dataset 2",
                options=df2.columns.tolist()
            )
            
            combine_method = st.radio(
                "Combine Method",
                ["best", "concatenate"],
                horizontal=True,
                help="'best' tries each column separately, 'concatenate' joins columns before matching"
            )
            
            if df1_cols and st.button("🚀 Run Matching", type="primary"):
                with st.spinner("Performing fuzzy matching..."):
                    results = st.session_state.matcher.fuzzy_match_many_to_one(
                        "Dataset_1",
                        "Dataset_2",
                        df1_cols,
                        df2_col,
                        combine_method=combine_method
                    )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} matches!")
                        
                        # Display statistics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Matches", len(results))
                        col2.metric("Average Score", f"{results['match_score'].mean():.1f}%")
                        col3.metric("Min Score", f"{results['match_score'].min():.1f}%")
                        
                        # Display results
                        st.subheader("📊 Matching Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv,
                            file_name="fuzzy_match_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No matches found above the threshold. Try lowering the threshold.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Matching Engine:** {'RapidFuzz' if USING_RAPIDFUZZ else 'FuzzyWuzzy'}"
    )
    st.sidebar.info(
        "💡 **Tips:**\n"
        "- Start with 80% threshold\n"
        "- Use token_sort_ratio for most cases\n"
        "- Review matches manually\n"
        "- Adjust threshold if needed"
    )


if __name__ == "__main__":
    main()