import streamlit as st
import pandas as pd
import numpy as np
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Marketo Performance Auditor", layout="wide")

# Target columns for specific percentage formatting
PCT_COLS = ["% Delivered", "% Opened", "% Clicked Email", "Clicked to Open Ratio"]

# --- GLOBAL HELPERS ---

def reset_and_clear_cache():
    """Wipes session state. Streamlit will automatically rerun after this executes."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def find_col(df, possible_names):
    """Helper to find a column name in a dataframe based on keywords."""
    for col in df.columns:
        if any(name.lower() in col.lower() for name in possible_names):
            return col
    return None

def load_and_process(uploaded_file):
    """Loads, sanitizes headers, and aggressively strips summary rows."""
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # --- 1. AGGRESSIVE TOTAL ROW STRIPPING ---
        if not df.empty:
            # A) Keyword search across all columns for 'total' or 'grand'
            mask = df.astype(str).apply(lambda x: x.str.contains('total|summary|grand', case=False, na=False)).any(axis=1)
            
            # B) Look for 'NaN' in Activity columns (Total rows usually don't have dates)
            date_cols = [c for c in df.columns if 'activity' in c.lower()]
            if date_cols:
                mask = mask | df[date_cols[0]].isna()
                
            df = df[~mask].copy()

        # 2. Header Normalization
        clean_cols = []
        for col in df.columns:
            if any(x in col.lower() for x in ['click to open', 'ctor', 'clicked to open']):
                clean_cols.append("Clicked to Open Ratio")
            elif "% del" in col.lower():
                clean_cols.append("% Delivered")
            elif "% open" in col.lower():
                clean_cols.append("% Opened")
            elif "% click" in col.lower():
                clean_cols.append("% Clicked Email")
            else:
                new_name = re.sub(r'\s*\(.*\)', '', col)
                new_name = re.sub(r'\b(Email|Campaign)\b', '', new_name, flags=re.IGNORECASE).strip()
                clean_cols.append(new_name)
        df.columns = clean_cols

        # 3. Numeric Casting
        for col in df.columns:
            if col in PCT_COLS:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(r'[%\s,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col] is not None and df[col].max() > 1.0:
                    df[col] = df[col] / 100.0
            
            vol_keys = ['sent', 'delivered', 'opened', 'clicked', 'clicks', 'bounced']
            if any(k in col.lower() for k in vol_keys) and col not in PCT_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# --- MAIN APP FUNCTION ---

def main():
    st.title("📊 Marketo Performance Auditor")
    
    file = st.file_uploader("Upload Campaign Data (.csv, .xlsx)", type=["csv", "xlsx"])
    
    if file:
        if 'df' not in st.session_state or st.session_state.get('fn') != file.name:
            st.session_state['df'] = load_and_process(file)
            st.session_state['fn'] = file.name
        df = st.session_state['df']

        st.sidebar.header("Control Panel")
        st.sidebar.caption("App Version: v1.1.0")
        st.sidebar.button("Clear Cache & Reset Data", on_click=reset_and_clear_cache)
        
        # --- 1. GLOBAL FILTERS ---
        st.sidebar.subheader("1. Global Filters")
        filtered_df = df.copy()
        filter_cols = st.sidebar.multiselect("Select columns to filter", df.columns, key="filter_cols_list")
        
        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                c1, c2 = st.sidebar.columns(2)
                is_vol = any(k in col.lower() for k in ['sent', 'delivered', 'opened', 'clicked', 'clicks']) and col not in PCT_COLS
                
                if is_vol:
                    low = c1.number_input(f"{col} Min", value=int(df[col].min()), step=1, key=f"filter_min_{col}")
                    high = c2.number_input(f"{col} Max", value=int(df[col].max()), step=1, key=f"filter_max_{col}")
                else:
                    step_val = 0.001
                    low = c1.number_input(f"{col} Min", value=float(df[col].min()), step=step_val, format="%.3f", key=f"filter_min_{col}")
                    high = c2.number_input(f"{col} Max", value=float(df[col].max()), step=step_val, format="%.3f", key=f"filter_max_{col}")
                
                filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)]
            else:
                search_term = st.sidebar.text_input(f"Search {col}", key=f"search_input_{col}")
                all_items = sorted(df[col].unique().tolist())
                options = [item for item in all_items if search_term.lower() in str(item).lower()] if search_term else all_items
                
                if st.sidebar.button(f"Select All {len(options)} results", key=f"btn_{col}"):
                    st.session_state[f"filter_select_{col}"] = options

                selection = st.sidebar.multiselect(f"Filter {col}", options=options, key=f"filter_select_{col}")
                
                if selection:
                    filtered_df = filtered_df[filtered_df[col].isin(selection)]

        # --- 2. BENCHMARK SETTINGS ---
        st.sidebar.divider()
        st.sidebar.header("2. Benchmark Settings")
        use_bench = st.sidebar.checkbox("Enable Audit Highlighting", key="bench_active")
        isolate_results = st.sidebar.toggle("Only Show Data Matching Audit Criteria", key="iso_toggle")
        
        bench_criteria = {}
        if use_bench:
            logic_mode = st.sidebar.radio("Audit Logic Mode", ["Match ANY (OR)", "Match ALL (AND)"], key="logic_selector")
            targets = st.sidebar.multiselect("Audit Metrics", df.select_dtypes(include=[np.number]).columns, key="bench_targets")
            
            for t in targets:
                op = st.sidebar.selectbox(f"Condition for {t}", ["Select", "Less Than", "Greater Than"], key=f"bench_op_{t}")
                is_vol = any(k in t.lower() for k in ['sent', 'delivered', 'opened', 'clicked', 'clicks']) and t not in PCT_COLS
                val = st.sidebar.number_input(f"Threshold for {t}", value=0.000, step=0.001 if not is_vol else 1.0, format="%.3f" if not is_vol else "%.0f", key=f"val_{t}")
                if op != "Select":
                    bench_criteria[t] = (op, val)

            def check_audit_fail(row):
                if not bench_criteria: return False
                results = [(row[col] < threshold if op == "Less Than" else row[col] > threshold) for col, (op, threshold) in bench_criteria.items()]
                return all(results) if "Match ALL" in logic_mode else any(results)

            if isolate_results and bench_criteria:
                filtered_df = filtered_df[filtered_df.apply(check_audit_fail, axis=1)]

        # --- 3. COMPARE SEGMENTS ---
        st.sidebar.divider()
        st.sidebar.header("3. Compare Segments")
        use_compare = st.sidebar.checkbox("Enable Side-by-Side Comparison")

        if use_compare:
            c_col = st.sidebar.selectbox("Column to Compare", df.columns, index=0)
            term_a = st.sidebar.text_input("Segment A Search", key="term_a")
            term_b = st.sidebar.text_input("Segment B Search", key="term_b")

            if term_a and term_b:
                df_a = filtered_df[filtered_df[c_col].astype(str).str.contains(term_a, case=False, na=False)]
                df_b = filtered_df[filtered_df[c_col].astype(str).str.contains(term_b, case=False, na=False)]

                st.subheader(f"Comparison: '{term_a}' vs '{term_b}'")
                comp_1, comp_2 = st.columns(2)

                with comp_1:
                    st.info(f"Segment A: {term_a} ({len(df_a)} items)")
                    a_open = df_a["% Opened"].mean() if "% Opened" in df_a.columns else 0
                    a_ctor = df_a["Clicked to Open Ratio"].mean() if "Clicked to Open Ratio" in df_a.columns else 0
                    st.metric(f"{term_a} Open Rate", f"{a_open:.2%}")
                    st.metric(f"{term_a} CTOR", f"{a_ctor:.2%}")

                with comp_2:
                    st.info(f"Segment B: {term_b} ({len(df_b)} items)")
                    b_open = df_b["% Opened"].mean() if "% Opened" in df_b.columns else 0
                    b_ctor = df_b["Clicked to Open Ratio"].mean() if "Clicked to Open Ratio" in df_b.columns else 0
                    st.metric(f"{term_b} Open Rate", f"{b_open:.2%}", delta=f"{b_open - a_open:.2%}")
                    st.metric(f"{term_b} CTOR", f"{b_ctor:.2%}", delta=f"{b_ctor - a_ctor:.2%}")
                st.divider()

        # --- 4. CUSTOM SUMMARY METRICS ---
        # --- 4. CUSTOM SUMMARY METRICS ---
        if not filtered_df.empty:
            st.divider()
            
            st.sidebar.subheader("4. Summary Display")
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # --- FIXED DEFAULTS ---
            # We look for the exact normalized names we created during load
            default_selection = [
                col for col in ["Sent", "Delivered", "% Opened", "Clicked to Open Ratio"] 
                if col in numeric_cols
            ]
            
            selected_metrics = st.sidebar.multiselect(
                "Metrics to summarize", 
                options=numeric_cols, 
                default=default_selection,
                key="summary_metric_choice"
            )

            active_searches = [st.session_state[k] for k in st.session_state 
                               if k.startswith("search_input_") and st.session_state[k]]
            search_context = f" matching '{', '.join(active_searches)}'" if active_searches else ""
            st.subheader(f"Summary for {len(filtered_df)} Results{search_context}")

            if selected_metrics:
                rows = [selected_metrics[i:i + 4] for i in range(0, len(selected_metrics), 4)]
                
                # Fetch denominators for weighted math
                sent_c = find_col(filtered_df, ['sent'])
                del_c = find_col(filtered_df, ['delivered'])
                open_c = find_col(filtered_df, ['opened'])

                for row in rows:
                    cols = st.columns(len(row))
                    for i, metric_name in enumerate(row):
                        with cols[i]:
                            if metric_name in PCT_COLS:
                                total_num, total_den = 0, 1
                                if "Click" in metric_name and open_c:
                                    total_num = filtered_df[find_col(filtered_df, ['click'])].sum()
                                    total_den = filtered_df[open_c].sum()
                                elif del_c:
                                    num_col = find_col(filtered_df, [metric_name.replace('%','').strip()])
                                    total_num = filtered_df[num_col].sum() if num_col else 0
                                    total_den = filtered_df[del_c].sum()
                                
                                val = total_num / total_den if total_den > 0 else 0
                                st.metric(f"Weighted {metric_name}", f"{val:.2%}")
                            else:
                                total_val = filtered_df[metric_name].sum()
                                st.metric(f"Total {metric_name}", f"{int(total_val):,}")
            st.divider()

            # --- DATA TABLE VIEW ---
            col_config = {pct_col: st.column_config.NumberColumn(pct_col, format="%.2f%%") for pct_col in PCT_COLS if pct_col in filtered_df.columns}

            def auditor_style(row):
                if use_bench:
                    fail = check_audit_fail(row)
                    return ['background-color: rgba(255, 75, 75, 0.4)' if fail else '' for _ in row]
                return ['' for _ in row]

            st.dataframe(filtered_df.style.apply(auditor_style, axis=1), width="stretch", column_config=col_config)
            
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Export Filtered Results", data=csv_data, file_name="audited_data.csv", mime="text/csv")
        else:
            st.warning("No records match your filters or audit criteria.")

if __name__ == "__main__":
    main()