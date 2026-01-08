"""
Fuzzy Matching Streamlit Application
A web-based interface for performing fuzzy matching between CSV files.

Installation:
pip install streamlit pandas rapidfuzz

Run with:
streamlit run fuzzy_match_app_enhanced.py
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
    
    def fuzzy_match_two_columns(self,
                                df1_name: str,
                                df2_name: str,
                                df1_col1: str,
                                df1_col2: str,
                                df2_col1: str,
                                df2_col2: str,
                                method: str = 'token_sort_ratio',
                                weight_col1: float = 0.6,
                                weight_col2: float = 0.4,
                                require_both: bool = False) -> pd.DataFrame:
        """
        Perform fuzzy matching between two DataFrames using two columns from each.
        
        Parameters:
        - df1_name, df2_name: Names of the dataframes
        - df1_col1, df1_col2: Column names from dataset 1
        - df2_col1, df2_col2: Column names from dataset 2
        - method: Fuzzy matching method to use
        - weight_col1: Weight for first column (0-1)
        - weight_col2: Weight for second column (0-1)
        - require_both: If True, both columns must match above threshold
        """
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
        df1[df1_col1] = df1[df1_col1].fillna('').astype(str)
        df1[df1_col2] = df1[df1_col2].fillna('').astype(str)
        df2[df2_col1] = df2[df2_col1].fillna('').astype(str)
        df2[df2_col2] = df2[df2_col2].fillna('').astype(str)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df1)
        
        for idx1, row1 in df1.iterrows():
            progress = (idx1 + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx1 + 1} of {total_rows}...")
            
            search_value1 = row1[df1_col1]
            search_value2 = row1[df1_col2]
            
            # Skip if both values are empty
            if search_value1.strip() == '' and search_value2.strip() == '':
                continue
            
            best_score = 0
            best_match_idx = None
            best_match_scores = None
            
            # Compare against each row in df2
            for idx2, row2 in df2.iterrows():
                match_value1 = row2[df2_col1]
                match_value2 = row2[df2_col2]
                
                # Calculate scores for each column
                score1 = 0
                score2 = 0
                
                if search_value1.strip() != '' and match_value1.strip() != '':
                    score1 = fuzzy_func(search_value1, match_value1)
                
                if search_value2.strip() != '' and match_value2.strip() != '':
                    score2 = fuzzy_func(search_value2, match_value2)
                
                # Calculate combined score
                if require_both:
                    # Both columns must meet threshold
                    if score1 >= self.threshold and score2 >= self.threshold:
                        combined_score = (score1 * weight_col1) + (score2 * weight_col2)
                    else:
                        combined_score = 0
                else:
                    # Use weighted average
                    combined_score = (score1 * weight_col1) + (score2 * weight_col2)
                
                # Update best match if this is better
                if combined_score > best_score:
                    best_score = combined_score
                    best_match_idx = idx2
                    best_match_scores = (score1, score2)
            
            # If we found a match above threshold, add it to results
            if best_score >= self.threshold and best_match_idx is not None:
                matched_row = df2.iloc[best_match_idx]
                
                result = {
                    f'{df1_name}_index': idx1,
                    f'{df2_name}_index': best_match_idx,
                    'combined_score': round(best_score, 2),
                    f'{df1_col1}_score': round(best_match_scores[0], 2),
                    f'{df1_col2}_score': round(best_match_scores[1], 2),
                    f'{df1_name}_{df1_col1}': search_value1,
                    f'{df1_name}_{df1_col2}': search_value2,
                    f'{df2_name}_{df2_col1}': matched_row[df2_col1],
                    f'{df2_name}_{df2_col2}': matched_row[df2_col2]
                }
                
                # Add remaining columns from df1
                for col in df1.columns:
                    if col not in [df1_col1, df1_col2]:
                        result[f'{df1_name}_{col}'] = row1[col]
                
                # Add remaining columns from df2
                for col in df2.columns:
                    if col not in [df2_col1, df2_col2]:
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
            ["Single Column Match", "Two Column Match", "Multiple Columns to One"],
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
        
        elif match_type == "Two Column Match":
            st.markdown("**Match two columns from each dataset (e.g., Address + Expiration Date)**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset 1 Columns**")
                df1_col1 = st.selectbox(
                    "First column from Dataset 1",
                    options=df1.columns.tolist(),
                    key="df1_col1"
                )
                df1_col2 = st.selectbox(
                    "Second column from Dataset 1",
                    options=df1.columns.tolist(),
                    key="df1_col2"
                )
            
            with col2:
                st.markdown("**Dataset 2 Columns**")
                df2_col1 = st.selectbox(
                    "First column from Dataset 2",
                    options=df2.columns.tolist(),
                    key="df2_col1"
                )
                df2_col2 = st.selectbox(
                    "Second column from Dataset 2",
                    options=df2.columns.tolist(),
                    key="df2_col2"
                )
            
            st.markdown("**Matching Parameters**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weight_col1 = st.slider(
                    f"Weight for column 1",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.1,
                    help="How much to weight the first column (0-1)"
                )
            
            with col2:
                weight_col2 = st.slider(
                    f"Weight for column 2",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.1,
                    help="How much to weight the second column (0-1)"
                )
            
            with col3:
                require_both = st.checkbox(
                    "Require both columns to match",
                    value=False,
                    help="If checked, both columns must individually meet the threshold"
                )
            
            # Normalize weights
            total_weight = weight_col1 + weight_col2
            if total_weight > 0:
                weight_col1_norm = weight_col1 / total_weight
                weight_col2_norm = weight_col2 / total_weight
            else:
                weight_col1_norm = 0.5
                weight_col2_norm = 0.5
            
            st.info(f"ℹ️ Normalized weights: Column 1 = {weight_col1_norm:.1%}, Column 2 = {weight_col2_norm:.1%}")
            
            if st.button("🚀 Run Matching", type="primary"):
                with st.spinner("Performing two-column fuzzy matching..."):
                    results = st.session_state.matcher.fuzzy_match_two_columns(
                        "Dataset_1",
                        "Dataset_2",
                        df1_col1,
                        df1_col2,
                        df2_col1,
                        df2_col2,
                        method=matching_method,
                        weight_col1=weight_col1_norm,
                        weight_col2=weight_col2_norm,
                        require_both=require_both
                    )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} matches!")
                        
                        # Display statistics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Matches", len(results))
                        col2.metric("Avg Combined Score", f"{results['combined_score'].mean():.1f}%")
                        col3.metric("Avg Col1 Score", f"{results[f'{df1_col1}_score'].mean():.1f}%")
                        col4.metric("Avg Col2 Score", f"{results[f'{df1_col2}_score'].mean():.1f}%")
                        
                        # Display results
                        st.subheader("📊 Matching Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv,
                            file_name="fuzzy_match_two_column_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No matches found above the threshold. Try lowering the threshold or adjusting weights.")
        
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
        "- For two-column matching, weight the more unique column higher\n"
        "- Review matches manually\n"
        "- Adjust threshold if needed"
    )


if __name__ == "__main__":
    main()