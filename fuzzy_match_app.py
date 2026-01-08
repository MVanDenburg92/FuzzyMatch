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
    
    def fuzzy_match_flexible_columns(self,
                                     df1_name: str,
                                     df2_name: str,
                                     df1_cols: List[str],
                                     df2_cols: List[str],
                                     method: str = 'token_sort_ratio',
                                     weights: List[float] = None,
                                     require_all: bool = False) -> pd.DataFrame:
        """
        Perform fuzzy matching between two DataFrames using flexible number of columns (1-5 from each).
        
        Parameters:
        - df1_name, df2_name: Names of the dataframes
        - df1_cols: List of 1-5 column names from dataset 1
        - df2_cols: List of 1-5 column names from dataset 2
        - method: Fuzzy matching method to use
        - weights: List of weights for each column pair (will be normalized)
        - require_all: If True, all columns must match above threshold
        """
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        num_cols = max(len(df1_cols), len(df2_cols))
        
        # Pad shorter list with empty strings
        while len(df1_cols) < num_cols:
            df1_cols.append('')
        while len(df2_cols) < num_cols:
            df2_cols.append('')
        
        # Default weights if not provided
        if weights is None or len(weights) == 0:
            weights = [1.0] * num_cols
        else:
            # Pad weights if needed
            while len(weights) < num_cols:
                weights.append(0.0)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / num_cols] * num_cols
        
        # Choose fuzzy matching method
        method_map = {
            'ratio': fuzz.ratio,
            'partial_ratio': fuzz.partial_ratio,
            'token_sort_ratio': fuzz.token_sort_ratio,
            'token_set_ratio': fuzz.token_set_ratio
        }
        
        fuzzy_func = method_map.get(method, fuzz.token_sort_ratio)
        
        results = []
        
        # Clean data - only for non-empty columns
        for col in df1_cols:
            if col:
                df1[col] = df1[col].fillna('').astype(str)
        for col in df2_cols:
            if col:
                df2[col] = df2[col].fillna('').astype(str)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df1)
        
        for idx1, row1 in df1.iterrows():
            progress = (idx1 + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx1 + 1} of {total_rows}...")
            
            # Get search values (empty string for missing columns)
            search_values = []
            for col in df1_cols:
                if col:
                    search_values.append(row1[col])
                else:
                    search_values.append('')
            
            # Skip if all values are empty
            if all(val.strip() == '' for val in search_values):
                continue
            
            best_score = 0
            best_match_idx = None
            best_match_scores = None
            
            # Compare against each row in df2
            for idx2, row2 in df2.iterrows():
                match_values = []
                for col in df2_cols:
                    if col:
                        match_values.append(row2[col])
                    else:
                        match_values.append('')
                
                # Calculate scores for each column pair
                scores = []
                valid_comparisons = 0
                
                for i in range(num_cols):
                    # Only calculate score if both columns are specified and have values
                    if (df1_cols[i] and df2_cols[i] and 
                        search_values[i].strip() != '' and match_values[i].strip() != ''):
                        score = fuzzy_func(search_values[i], match_values[i])
                        scores.append(score)
                        valid_comparisons += 1
                    else:
                        scores.append(0)
                
                # Calculate combined score only if we have valid comparisons
                if valid_comparisons > 0:
                    if require_all:
                        # All specified columns must meet threshold
                        valid_scores = [scores[i] for i in range(num_cols) 
                                       if df1_cols[i] and df2_cols[i] and 
                                       search_values[i].strip() != '' and match_values[i].strip() != '']
                        if all(score >= self.threshold for score in valid_scores):
                            combined_score = sum(scores[i] * weights[i] for i in range(num_cols))
                        else:
                            combined_score = 0
                    else:
                        # Use weighted average
                        combined_score = sum(scores[i] * weights[i] for i in range(num_cols))
                else:
                    combined_score = 0
                
                # Update best match if this is better
                if combined_score > best_score:
                    best_score = combined_score
                    best_match_idx = idx2
                    best_match_scores = scores
            
            # If we found a match above threshold, add it to results
            if best_score >= self.threshold and best_match_idx is not None:
                matched_row = df2.iloc[best_match_idx]
                
                result = {
                    f'{df1_name}_index': idx1,
                    f'{df2_name}_index': best_match_idx,
                    'combined_score': round(best_score, 2)
                }
                
                # Add individual scores and matched values for specified columns
                for i in range(num_cols):
                    if df1_cols[i] and df2_cols[i]:
                        result[f'col{i+1}_score'] = round(best_match_scores[i], 2)
                        result[f'{df1_name}_{df1_cols[i]}'] = search_values[i]
                        result[f'{df2_name}_{df2_cols[i]}'] = matched_row[df2_cols[i]]
                
                # Add remaining columns from df1
                for col in df1.columns:
                    if col not in [c for c in df1_cols if c]:
                        result[f'{df1_name}_{col}'] = row1[col]
                
                # Add remaining columns from df2
                for col in df2.columns:
                    if col not in [c for c in df2_cols if c]:
                        result[f'{df2_name}_{col}'] = matched_row[col]
                
                results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    
    def create_left_join(self, df1_name: str, df2_name: str, match_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create a left join: All records from dataset 1 with their matches from dataset 2.
        Records without matches will have NaN for dataset 2 columns.
        """
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        # Add prefix to all df1 columns
        df1_renamed = df1.add_prefix(f'{df1_name}_')
        df1_renamed[f'{df1_name}_index'] = df1.index
        
        # Add prefix to all df2 columns
        df2_renamed = df2.add_prefix(f'{df2_name}_')
        df2_renamed[f'{df2_name}_index'] = df2.index
        
        # Create a simplified match lookup
        if len(match_results) > 0:
            match_lookup = match_results[[f'{df1_name}_index', f'{df2_name}_index', 'combined_score']].copy()
            
            # Merge df1 with match results
            result = df1_renamed.merge(
                match_lookup,
                on=f'{df1_name}_index',
                how='left'
            )
            
            # Merge with df2 data
            result = result.merge(
                df2_renamed,
                on=f'{df2_name}_index',
                how='left'
            )
        else:
            # No matches, just return df1 with empty df2 columns
            result = df1_renamed.copy()
            result['combined_score'] = None
            for col in df2.columns:
                result[f'{df2_name}_{col}'] = None
        
        return result
    
    def create_right_join(self, df1_name: str, df2_name: str, match_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create a right join: All records from dataset 2 with their matches from dataset 1.
        Records without matches will have NaN for dataset 1 columns.
        """
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        # Add prefix to all df1 columns
        df1_renamed = df1.add_prefix(f'{df1_name}_')
        df1_renamed[f'{df1_name}_index'] = df1.index
        
        # Add prefix to all df2 columns
        df2_renamed = df2.add_prefix(f'{df2_name}_')
        df2_renamed[f'{df2_name}_index'] = df2.index
        
        # Create a simplified match lookup
        if len(match_results) > 0:
            match_lookup = match_results[[f'{df1_name}_index', f'{df2_name}_index', 'combined_score']].copy()
            
            # Merge df2 with match results
            result = df2_renamed.merge(
                match_lookup,
                on=f'{df2_name}_index',
                how='left'
            )
            
            # Merge with df1 data
            result = result.merge(
                df1_renamed,
                on=f'{df1_name}_index',
                how='left'
            )
        else:
            # No matches, just return df2 with empty df1 columns
            result = df2_renamed.copy()
            result['combined_score'] = None
            for col in df1.columns:
                result[f'{df1_name}_{col}'] = None
        
        return result
    
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
            ["Single Column Match", "Multi-Column Match (1-5 columns)", "Multiple Columns to One"],
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
                    
                    # Store results in session state
                    st.session_state.match_results = results
                    
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
                        
                        # Download section
                        st.subheader("⬇️ Download Options")
                        
                        download_cols = st.columns(3)
                        
                        with download_cols[0]:
                            # Download matches only
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📄 Download Matches Only",
                                data=csv,
                                file_name="fuzzy_match_results.csv",
                                mime="text/csv",
                                help="Download only the matched records"
                            )
                        
                        with download_cols[1]:
                            # Left Join - All Dataset 1 + Matches from Dataset 2
                            left_join = st.session_state.matcher.create_left_join(
                                "Dataset_1", "Dataset_2", results
                            )
                            csv_left = left_join.to_csv(index=False)
                            st.download_button(
                                label="⬅️ Left Join (All Dataset 1)",
                                data=csv_left,
                                file_name="left_join_all_dataset1_plus_matches.csv",
                                mime="text/csv",
                                help="All records from Dataset 1 with matching records from Dataset 2"
                            )
                        
                        with download_cols[2]:
                            # Right Join - All Dataset 2 + Matches from Dataset 1
                            right_join = st.session_state.matcher.create_right_join(
                                "Dataset_1", "Dataset_2", results
                            )
                            csv_right = right_join.to_csv(index=False)
                            st.download_button(
                                label="➡️ Right Join (All Dataset 2)",
                                data=csv_right,
                                file_name="right_join_all_dataset2_plus_matches.csv",
                                mime="text/csv",
                                help="All records from Dataset 2 with matching records from Dataset 1"
                            )
                    else:
                        st.warning("⚠️ No matches found above the threshold. Try lowering the threshold.")
        
        elif match_type == "Multi-Column Match (1-5 columns)":
            st.markdown("**Match using 1-5 columns from each dataset. Each dataset can have a different number of columns.**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset 1 Columns**")
                num_cols_df1 = st.number_input(
                    "Number of columns from Dataset 1",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="num_cols_df1"
                )
                
                df1_cols = []
                for i in range(int(num_cols_df1)):
                    col = st.selectbox(
                        f"Column {i+1} from Dataset 1",
                        options=df1.columns.tolist(),
                        key=f"df1_col{i}",
                        help="Select a column to match"
                    )
                    df1_cols.append(col)
            
            with col2:
                st.markdown("**Dataset 2 Columns**")
                num_cols_df2 = st.number_input(
                    "Number of columns from Dataset 2",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="num_cols_df2"
                )
                
                df2_cols = []
                for i in range(int(num_cols_df2)):
                    col = st.selectbox(
                        f"Column {i+1} from Dataset 2",
                        options=df2.columns.tolist(),
                        key=f"df2_col{i}",
                        help="Select a column to match"
                    )
                    df2_cols.append(col)
            
            # Determine the maximum number of columns
            max_cols = max(len(df1_cols), len(df2_cols))
            
            st.markdown("---")
            st.markdown("**Matching Parameters**")
            
            # Show column pairing information
            st.info(f"ℹ️ Dataset 1 has {len(df1_cols)} column(s), Dataset 2 has {len(df2_cols)} column(s). "
                   f"Matching will compare up to {max_cols} column pair(s).")
            
            # Create weight sliders for each column pair
            st.markdown("**Column Weights** (adjust importance of each column pair)")
            
            # Create columns for weight sliders
            weight_cols = st.columns(min(max_cols, 5))
            weights = []
            
            for i in range(max_cols):
                with weight_cols[i % 5]:
                    # Show which columns are being compared
                    df1_col_name = df1_cols[i] if i < len(df1_cols) else "None"
                    df2_col_name = df2_cols[i] if i < len(df2_cols) else "None"
                    
                    weight = st.slider(
                        f"Pair {i+1}\n({df1_col_name} ↔ {df2_col_name})",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / max_cols,
                        step=0.05,
                        key=f"weight{i}",
                        help=f"Weight for comparing {df1_col_name} with {df2_col_name}"
                    )
                    weights.append(weight)
            
            # Normalize and display weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                weight_display = " | ".join([f"Pair{i+1}: {w:.1%}" for i, w in enumerate(normalized_weights)])
                st.success(f"✓ Normalized weights: {weight_display}")
            else:
                normalized_weights = [1.0 / max_cols] * max_cols
                st.warning("⚠️ All weights are 0, using equal weights")
            
            require_all = st.checkbox(
                "Require all column pairs to match individually",
                value=False,
                help="If checked, all specified column pairs must individually meet the threshold"
            )
            
            if st.button("🚀 Run Matching", type="primary"):
                with st.spinner(f"Performing multi-column fuzzy matching with {max_cols} column pair(s)..."):
                    results = st.session_state.matcher.fuzzy_match_flexible_columns(
                        "Dataset_1",
                        "Dataset_2",
                        df1_cols,
                        df2_cols,
                        method=matching_method,
                        weights=weights,
                        require_all=require_all
                    )
                    
                    # Store results in session state
                    st.session_state.match_results = results
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} matches!")
                        
                        # Display statistics - show combined score plus individual column scores
                        num_stat_cols = min(max_cols + 1, 6)  # Combined + up to 5 column scores
                        stat_cols = st.columns(num_stat_cols)
                        
                        stat_cols[0].metric("Total Matches", len(results))
                        
                        if 'combined_score' in results.columns:
                            stat_cols[0].metric("Avg Combined", f"{results['combined_score'].mean():.1f}%")
                        
                        # Show average scores for each column pair
                        col_idx = 1
                        for i in range(1, max_cols + 1):
                            if f'col{i}_score' in results.columns and col_idx < num_stat_cols:
                                stat_cols[col_idx].metric(
                                    f"Avg Pair{i}", 
                                    f"{results[f'col{i}_score'].mean():.1f}%"
                                )
                                col_idx += 1
                        
                        # Display results
                        st.subheader("📊 Matching Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Download section
                        st.subheader("⬇️ Download Options")
                        
                        download_cols = st.columns(3)
                        
                        with download_cols[0]:
                            # Download matches only
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📄 Download Matches Only",
                                data=csv,
                                file_name="fuzzy_match_multi_column_results.csv",
                                mime="text/csv",
                                help="Download only the matched records"
                            )
                        
                        with download_cols[1]:
                            # Left Join - All Dataset 1 + Matches from Dataset 2
                            left_join = st.session_state.matcher.create_left_join(
                                "Dataset_1", "Dataset_2", results
                            )
                            csv_left = left_join.to_csv(index=False)
                            st.download_button(
                                label="⬅️ Left Join (All Dataset 1)",
                                data=csv_left,
                                file_name="left_join_all_dataset1_plus_matches.csv",
                                mime="text/csv",
                                help="All records from Dataset 1 with matching records from Dataset 2"
                            )
                        
                        with download_cols[2]:
                            # Right Join - All Dataset 2 + Matches from Dataset 1
                            right_join = st.session_state.matcher.create_right_join(
                                "Dataset_1", "Dataset_2", results
                            )
                            csv_right = right_join.to_csv(index=False)
                            st.download_button(
                                label="➡️ Right Join (All Dataset 2)",
                                data=csv_right,
                                file_name="right_join_all_dataset2_plus_matches.csv",
                                mime="text/csv",
                                help="All records from Dataset 2 with matching records from Dataset 1"
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
                    
                    # Store results in session state
                    st.session_state.match_results = results
                    
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
                        
                        # Download section
                        st.subheader("⬇️ Download Options")
                        
                        download_cols = st.columns(3)
                        
                        with download_cols[0]:
                            # Download matches only
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📄 Download Matches Only",
                                data=csv,
                                file_name="fuzzy_match_results.csv",
                                mime="text/csv",
                                help="Download only the matched records"
                            )
                        
                        with download_cols[1]:
                            # Left Join - All Dataset 1 + Matches from Dataset 2
                            left_join = st.session_state.matcher.create_left_join(
                                "Dataset_1", "Dataset_2", results
                            )
                            csv_left = left_join.to_csv(index=False)
                            st.download_button(
                                label="⬅️ Left Join (All Dataset 1)",
                                data=csv_left,
                                file_name="left_join_all_dataset1_plus_matches.csv",
                                mime="text/csv",
                                help="All records from Dataset 1 with matching records from Dataset 2"
                            )
                        
                        with download_cols[2]:
                            # Right Join - All Dataset 2 + Matches from Dataset 1
                            right_join = st.session_state.matcher.create_right_join(
                                "Dataset_1", "Dataset_2", results
                            )
                            csv_right = right_join.to_csv(index=False)
                            st.download_button(
                                label="➡️ Right Join (All Dataset 2)",
                                data=csv_right,
                                file_name="right_join_all_dataset2_plus_matches.csv",
                                mime="text/csv",
                                help="All records from Dataset 2 with matching records from Dataset 1"
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
        "- For multi-column matching, weight the most unique columns higher\n"
        "- Each dataset can have 1-5 columns\n"
        "- Review matches manually\n"
        "- Adjust threshold if needed"
    )


if __name__ == "__main__":
    main()