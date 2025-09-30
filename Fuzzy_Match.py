"""
Fuzzy Matching Script for Multiple CSV Files
This script performs fuzzy matching between two or more CSV files based on user-specified columns.
Requires: pandas, fuzzywuzzy (or rapidfuzz), python-Levenshtein (optional but recommended for speed)

Install dependencies:
pip install pandas fuzzywuzzy python-Levenshtein
# OR for faster performance:
pip install pandas rapidfuzz
"""

import pandas as pd
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import rapidfuzz first (faster), fall back to fuzzywuzzy
USING_RAPIDFUZZ = False
try:
    from rapidfuzz import fuzz, process
    USING_RAPIDFUZZ = True
    print("Using rapidfuzz for fuzzy matching (faster)")
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        USING_RAPIDFUZZ = False
        print("Using fuzzywuzzy for fuzzy matching")
    except ImportError:
        raise ImportError(
            "Neither rapidfuzz nor fuzzywuzzy is installed. "
            "Please install one of them:\n"
            "  pip install rapidfuzz  (recommended, faster)\n"
            "  OR\n"
            "  pip install fuzzywuzzy python-Levenshtein"
        )


class FuzzyMatcher:
    """
    A class to perform fuzzy matching between multiple CSV files.
    """
    
    def __init__(self, threshold: int = 80):
        """
        Initialize the FuzzyMatcher.
        
        Args:
            threshold: Minimum similarity score (0-100) to consider a match. Default is 80.
        """
        self.threshold = threshold
        self.dataframes = {}
        
    def _extract_match_info(self, match_result):
        """
        Helper function to extract match information from either rapidfuzz or fuzzywuzzy results.
        
        Args:
            match_result: Result from process.extractOne()
            
        Returns:
            Tuple of (match_value, score, match_index)
        """
        if match_result is None:
            return None, 0, None
            
        if USING_RAPIDFUZZ:
            # rapidfuzz returns (value, score, index) as a tuple
            if len(match_result) == 3:
                return match_result[0], match_result[1], match_result[2]
            else:
                # Sometimes rapidfuzz returns just (value, score)
                return match_result[0], match_result[1], None
        else:
            # fuzzywuzzy returns (value, score, index) as a tuple
            if len(match_result) == 3:
                return match_result[0], match_result[1], match_result[2]
            else:
                return match_result[0], match_result[1], None
    
    def load_csv(self, file_path: str, name: str = None) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.
        
        Args:
            file_path: Path to the CSV file
            name: Optional name for the dataframe (defaults to filename)
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df_name = name if name else os.path.basename(file_path).replace('.csv', '')
        self.dataframes[df_name] = df
        
        print(f"Loaded '{df_name}': {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}\n")
        
        return df
    
    def fuzzy_match_single_column(self, 
                                   df1_name: str, 
                                   df2_name: str,
                                   col1: str, 
                                   col2: str,
                                   method: str = 'token_sort_ratio') -> pd.DataFrame:
        """
        Perform fuzzy matching between two DataFrames on single columns.
        
        Args:
            df1_name: Name of first dataframe
            df2_name: Name of second dataframe
            col1: Column name in first dataframe
            col2: Column name in second dataframe
            method: Matching method - 'ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'
            
        Returns:
            DataFrame with matching results
        """
        if df1_name not in self.dataframes or df2_name not in self.dataframes:
            raise ValueError("Dataframe names not found. Load CSVs first.")
        
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        if col1 not in df1.columns:
            raise ValueError(f"Column '{col1}' not found in {df1_name}")
        if col2 not in df2.columns:
            raise ValueError(f"Column '{col2}' not found in {df2_name}")
        
        # Choose fuzzy matching method
        method_map = {
            'ratio': fuzz.ratio,
            'partial_ratio': fuzz.partial_ratio,
            'token_sort_ratio': fuzz.token_sort_ratio,
            'token_set_ratio': fuzz.token_set_ratio
        }
        
        if method not in method_map:
            print(f"Unknown method '{method}', defaulting to 'token_sort_ratio'")
            method = 'token_sort_ratio'
        
        fuzzy_func = method_map[method]
        
        # Prepare results list
        results = []
        
        # Clean data - convert to string and handle nulls
        df1[col1] = df1[col1].fillna('').astype(str)
        df2[col2] = df2[col2].fillna('').astype(str)
        
        # Create list of values to match against
        choices = df2[col2].tolist()
        
        print(f"Matching {len(df1)} records from '{df1_name}' against {len(df2)} records from '{df2_name}'...")
        print(f"Using method: {method}\n")
        
        # Perform fuzzy matching for each record in df1
        for idx, row in df1.iterrows():
            search_value = row[col1]
            
            if search_value.strip() == '':
                continue
            
            # Find best match
            best_match = process.extractOne(search_value, choices, scorer=fuzzy_func)
            
            if best_match:
                match_value, score, match_idx = self._extract_match_info(best_match)
                
                if score >= self.threshold and match_idx is not None:
                
                # Get the matching row from df2
                    matched_row = df2.iloc[match_idx]
                    
                    # Create result record
                    result = {
                        f'{df1_name}_index': idx,
                        f'{df1_name}_{col1}': search_value,
                        f'{df2_name}_index': match_idx,
                        f'{df2_name}_{col2}': match_value,
                        'match_score': score
                    }
                    
                    # Add all columns from both dataframes
                    for col in df1.columns:
                        if col != col1:
                            result[f'{df1_name}_{col}'] = row[col]
                    
                    for col in df2.columns:
                        if col != col2:
                            result[f'{df2_name}_{col}'] = matched_row[col]
                    
                    results.append(result)
        
        results_df = pd.DataFrame(results)
        
        print(f"Found {len(results_df)} matches above threshold ({self.threshold})")
        print(f"Average match score: {results_df['match_score'].mean():.2f}\n")
        
        return results_df
    
    def fuzzy_match_many_to_one(self,
                                 df1_name: str,
                                 df2_name: str,
                                 df1_columns: List[str],
                                 df2_column: str,
                                 combine_method: str = 'best') -> pd.DataFrame:
        """
        Match multiple columns from df1 against a single column in df2.
        Useful for matching variations like first_name, last_name, nickname against a single name field.
        
        Args:
            df1_name: Name of first dataframe
            df2_name: Name of second dataframe
            df1_columns: List of columns from df1 to match
            df2_column: Single column from df2 to match against
            combine_method: 'best' (take highest score) or 'concatenate' (combine df1 columns first)
            
        Returns:
            DataFrame with matching results
        """
        if df1_name not in self.dataframes or df2_name not in self.dataframes:
            raise ValueError("Dataframe names not found. Load CSVs first.")
        
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        # Validate columns
        for col in df1_columns:
            if col not in df1.columns: 
                raise ValueError(f"Column '{col}' not found in {df1_name}")
        if df2_column not in df2.columns:
            raise ValueError(f"Column '{df2_column}' not found in {df2_name}")
        
        results = []
        
        print(f"Matching {len(df1_columns)} columns from '{df1_name}' against '{df2_column}' in '{df2_name}':")
        for col in df1_columns:
            print(f"  - {col}")
        print(f"Method: {combine_method}\n")
        
        # Clean data
        for col in df1_columns:
            df1[col] = df1[col].fillna('').astype(str)
        df2[df2_column] = df2[df2_column].fillna('').astype(str)
        
        # Create list of values to match against
        choices = df2[df2_column].tolist()
        
        for idx1, row1 in df1.iterrows():
            best_score = 0
            best_match_idx = None
            best_match_value = None
            matched_column = None
            column_scores = {}
            
            if combine_method == 'concatenate':
                # Concatenate all df1 columns and match as one string
                combined_value = ' '.join([row1[col] for col in df1_columns if row1[col].strip() != ''])
                
                if combined_value.strip():
                    match_result = process.extractOne(combined_value, choices, scorer=fuzz.token_sort_ratio)
                    
                    if match_result:
                        best_match_value, best_score, best_match_idx = self._extract_match_info(match_result)
                        
                        if best_score >= self.threshold and best_match_idx is not None:
                            matched_column = 'combined'
                        
            else:  # 'best' method - try each column and take best match
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
            
            # If best match exceeds threshold, add to results
            if best_score >= self.threshold and best_match_idx is not None:
                matched_row = df2.iloc[best_match_idx]
                
                result = {
                    f'{df1_name}_index': idx1,
                    f'{df2_name}_index': best_match_idx,
                    'match_score': best_score,
                    'matched_column': matched_column,
                    f'{df2_name}_{df2_column}': best_match_value
                }
                
                # Add all df1 columns that were searched
                for col in df1_columns:
                    result[f'{df1_name}_{col}'] = row1[col]
                    if combine_method == 'best' and col in column_scores:
                        result[f'score_{col}'] = column_scores[col]
                
                # Add remaining columns from both dataframes
                for col in df1.columns:
                    if col not in df1_columns:
                        result[f'{df1_name}_{col}'] = row1[col]
                
                for col in df2.columns:
                    if col != df2_column:
                        result[f'{df2_name}_{col}'] = matched_row[col]
                
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        print(f"Found {len(results_df)} matches above threshold ({self.threshold})")
        if len(results_df) > 0:
            print(f"Average match score: {results_df['match_score'].mean():.2f}\n")
        
        return results_df
    
    def fuzzy_match_multiple_columns(self,
                                      df1_name: str,
                                      df2_name: str,
                                      column_pairs: List[Tuple[str, str]],
                                      combine_method: str = 'average') -> pd.DataFrame:
        """
        Perform fuzzy matching on multiple column pairs and combine scores.
        
        Args:
            df1_name: Name of first dataframe
            df2_name: Name of second dataframe
            column_pairs: List of tuples (col1, col2) to match
            combine_method: How to combine scores - 'average', 'max', 'min', 'weighted'
            
        Returns:
            DataFrame with matching results
        """
        if df1_name not in self.dataframes or df2_name not in self.dataframes:
            raise ValueError("Dataframe names not found. Load CSVs first.")
        
        df1 = self.dataframes[df1_name].copy()
        df2 = self.dataframes[df2_name].copy()
        
        results = []
        
        print(f"Matching on {len(column_pairs)} column pairs:")
        for col1, col2 in column_pairs:
            print(f"  - {col1} <-> {col2}")
        print()
        
        # Clean data
        for col1, col2 in column_pairs:
            df1[col1] = df1[col1].fillna('').astype(str)
            df2[col2] = df2[col2].fillna('').astype(str)
        
        # Match each row in df1 against all rows in df2
        for idx1, row1 in df1.iterrows():
            best_match_idx = None
            best_combined_score = 0
            column_scores = []
            
            for idx2, row2 in df2.iterrows():
                scores = []
                
                # Calculate score for each column pair
                for col1, col2 in column_pairs:
                    val1 = row1[col1]
                    val2 = row2[col2]
                    
                    if val1.strip() == '' or val2.strip() == '':
                        score = 0
                    else:
                        score = fuzz.token_sort_ratio(val1, val2)
                    
                    scores.append(score)
                
                # Combine scores based on method
                if combine_method == 'average':
                    combined_score = sum(scores) / len(scores)
                elif combine_method == 'max':
                    combined_score = max(scores)
                elif combine_method == 'min':
                    combined_score = min(scores)
                else:  # weighted - first column gets more weight
                    weights = [1.0 / (i + 1) for i in range(len(scores))]
                    combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_match_idx = idx2
                    column_scores = scores
            
            # If best match exceeds threshold, add to results
            if best_combined_score >= self.threshold:
                matched_row = df2.iloc[best_match_idx]
                
                result = {
                    f'{df1_name}_index': idx1,
                    f'{df2_name}_index': best_match_idx,
                    'combined_score': best_combined_score
                }
                
                # Add individual column scores
                for i, (col1, col2) in enumerate(column_pairs):
                    result[f'score_{col1}_{col2}'] = column_scores[i]
                    result[f'{df1_name}_{col1}'] = row1[col1]
                    result[f'{df2_name}_{col2}'] = matched_row[col2]
                
                # Add remaining columns
                for col in df1.columns:
                    if col not in [c[0] for c in column_pairs]:
                        result[f'{df1_name}_{col}'] = row1[col]
                
                for col in df2.columns:
                    if col not in [c[1] for c in column_pairs]:
                        result[f'{df2_name}_{col}'] = matched_row[col]
                
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        print(f"Found {len(results_df)} matches above threshold ({self.threshold})")
        if len(results_df) > 0:
            print(f"Average combined score: {results_df['combined_score'].mean():.2f}\n")
        
        return results_df
    
    def export_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Export results to CSV.
        
        Args:
            results_df: DataFrame with matching results
            output_path: Path for output CSV file
        """
        results_df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")


def main():
    """
    Example usage of the FuzzyMatcher class.
    """
    print("=" * 80)
    print("FUZZY MATCHING SCRIPT FOR MULTIPLE CSVs")
    print("=" * 80)
    print()
    
    # Initialize matcher with 80% threshold
    matcher = FuzzyMatcher(threshold=80)
    
    # Example 1: Load CSVs
    print("STEP 1: Load CSV files")
    print("-" * 80)
    
    # Replace these with your actual file paths
    csv1_path = "file1.csv"
    csv2_path = "file2.csv"
    
    # Check if example files exist
    if not os.path.exists(csv1_path):
        print(f"Example files not found. Please update the paths in the script.")
        print(f"Looking for: {csv1_path} and {csv2_path}")
        print()
        print("Usage example:")
        print("  matcher = FuzzyMatcher(threshold=80)")
        print("  matcher.load_csv('customers.csv', 'customers')")
        print("  matcher.load_csv('vendors.csv', 'vendors')")
        print("  results = matcher.fuzzy_match_single_column('customers', 'vendors', 'company_name', 'vendor_name')")
        print("  matcher.export_results(results, 'matched_results.csv')")
        return
    
    # Load the CSVs
    df1 = matcher.load_csv(csv1_path, "dataset1")
    df2 = matcher.load_csv(csv2_path, "dataset2")
    
    # Example 2: Single column matching
    print("\nSTEP 2: Perform fuzzy matching on single columns")
    print("-" * 80)
    
    # Update these column names to match your data
    col1_name = "name"  # Column from first CSV
    col2_name = "company_name"  # Column from second CSV
    
    results_single = matcher.fuzzy_match_single_column(
        "dataset1", 
        "dataset2",
        col1_name,
        col2_name,
        method='token_sort_ratio'
    )
    
    # Example 3: Many-to-one matching (NEW FEATURE)
    print("\nSTEP 3: Match multiple columns to single column")
    print("-" * 80)
    
    # Example: Match first_name, last_name, nickname from dataset1 
    # against a single full_name column in dataset2
    results_many_to_one_best = matcher.fuzzy_match_many_to_one(
        "dataset1",
        "dataset2",
        df1_columns=["first_name", "last_name", "nickname"],  # Multiple columns
        df2_column="full_name",  # Single column
        combine_method='best'  # Take the best match from any column
    )
    
    # Alternative: Concatenate columns first, then match
    results_many_to_one_concat = matcher.fuzzy_match_many_to_one(
        "dataset1",
        "dataset2",
        df1_columns=["first_name", "last_name"],  # Combine these
        df2_column="full_name",
        combine_method='concatenate'  # Join columns before matching
    )
    
    # Example 4: Multiple column matching (original feature)
    print("\nSTEP 4: Perform fuzzy matching on multiple column pairs")
    print("-" * 80)
    
    # Define column pairs to match
    column_pairs = [
        ("name", "company_name"),
        ("address", "address"),
        ("city", "city")
    ]
    
    results_multi = matcher.fuzzy_match_multiple_columns(
        "dataset1",
        "dataset2",
        column_pairs,
        combine_method='average'
    )
    
    # Example 5: Export results
    print("\nSTEP 5: Export results")
    print("-" * 80)
    
    matcher.export_results(results_single, "fuzzy_match_single_column.csv")
    matcher.export_results(results_many_to_one_best, "fuzzy_match_many_to_one.csv")
    matcher.export_results(results_multi, "fuzzy_match_multiple_columns.csv")
    
    print("\nDone!")


if __name__ == "__main__":
    main()