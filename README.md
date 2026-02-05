Marketo Performance Auditor
A streamlined web application for auditing Marketo email campaign performance, comparing segments, and identifying engagement outliers.

Data Requirements
To ensure accurate audits, your Marketo export must:

Be in .csv or .xlsx format.

Include the following standard columns: Email Name, Sent, Delivered, Opened, and Clicked Email.

Note: The app automatically handles and removes the "Grand Total" summary rows often found at the bottom of Marketo exports.

Features
Dynamic Filtering: Search for specific program IDs (e.g., "1008") or campaign themes.

Side-by-Side Comparison: Pit two segments (e.g., "Newsletters" vs "Direct Sends") against each other to see performance deltas.

Weighted Metrics: Calculations for Open Rate and CTOR are weighted against total volume for mathematical accuracy.

Audit Highlighting: Define custom benchmarks to instantly flag underperforming campaigns in red.

Local Setup
Clone this repository.

Install dependencies: pip install -r requirements.txt

Run the app: streamlit run app.py