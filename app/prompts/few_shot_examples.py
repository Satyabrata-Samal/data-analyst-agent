"""Few-shot examples injected into LLM calls for code generation and critique nodes.

``CODE_GENERATOR_EXAMPLES`` and ``CRITIC_EXAMPLES`` are lists of alternating
user/assistant message dicts (``{"role": "user"|"assistant", "content": "..."}``)
used to demonstrate expected input/output format before the model handles the
live request.
"""

CODE_GENERATOR_EXAMPLES: list[dict] = [
    {
        "role": "user",
        "content": (
            "Dataset profile summary: Sales data with columns: region (object), "
            "revenue (float), date (object), product (object). "
            "1200 rows, no missing values.\n\n"
            "Analysis plan:\n"
            "Step 1: Group by region and sum revenue\n"
            "Step 2: Sort descending by total revenue\n"
            "Step 3: Print top regions with their totals\n\n"
            "User question: What are the top performing regions by revenue?"
        ),
    },
    {
        "role": "assistant",
        "content": """{
  "code": "region_revenue = df.groupby('region')['revenue'].sum().sort_values(ascending=False)\\nprint('=== Revenue by Region ===')\\nprint(region_revenue.to_string())\\nprint(f'\\\\nTop region: {region_revenue.index[0]} with revenue {region_revenue.iloc[0]:,.2f}')\\nprint(f'Total regions analyzed: {len(region_revenue)}')",
  "explanation": "Groups the dataframe by region, sums revenue for each, sorts descending, and prints a ranked table with the top region highlighted.",
  "expected_output_description": "A ranked table of regions with their total revenue, followed by a summary line identifying the top region and its revenue total."
}""",
    },
    {
        "role": "user",
        "content": (
            "Dataset profile summary: Employee data with columns: department (object), "
            "salary (float, 15% missing), tenure_years (float).\n"
            "800 rows.\n\n"
            "Analysis plan:\n"
            "Step 1: Drop rows where salary is null\n"
            "Step 2: Group by department, compute mean and median salary\n"
            "Step 3: Print comparison table\n\n"
            "User question: What is the average salary by department?"
        ),
    },
    {
        "role": "assistant",
        "content": """{
  "code": "df_clean = df.dropna(subset=['salary'])\\nprint(f'Rows after dropping null salaries: {len(df_clean)} (removed {len(df) - len(df_clean)})')\\nsalary_stats = df_clean.groupby('department')['salary'].agg(['mean', 'median', 'count'])\\nsalary_stats.columns = ['mean_salary', 'median_salary', 'employee_count']\\nsalary_stats = salary_stats.sort_values('mean_salary', ascending=False)\\nprint('\\\\n=== Salary by Department ===')\\nprint(salary_stats.round(2).to_string())",
  "explanation": "Drops rows with missing salary, then groups by department to compute mean, median, and count. Sorts by mean salary descending.",
  "expected_output_description": "A line showing how many rows were removed, followed by a table with department, mean salary, median salary, and employee count."
}""",
    },
]

CRITIC_EXAMPLES: list[dict] = [
    {
        "role": "user",
        "content": (
            "User question: What are the top performing regions by revenue?\n\n"
            "Current insights:\n"
            '- Title: "Regional Revenue"\n'
            '  Finding: "Some regions perform better than others."\n'
            "  Confidence: high"
        ),
    },
    {
        "role": "assistant",
        "content": """{
  "score": 3.5,
  "issues": [
    "Finding is too vague — 'some regions perform better' is not an insight, it is a truism",
    "No specific regions are named or ranked",
    "No revenue figures are cited despite being available in the output",
    "Confidence marked as high but finding contains no evidence"
  ],
  "suggestions": [
    "Name the top 3 regions with their actual revenue figures",
    "Include the revenue gap between top and bottom region",
    "Lower confidence to medium if full dataset was not analyzed"
  ]
}""",
    },
    {
        "role": "user",
        "content": (
            "User question: What is the average salary by department?\n\n"
            "Current insights:\n"
            '- Title: "Engineering Leads Compensation"\n'
            '  Finding: "Engineering department has the highest mean salary at $142,000, '
            "followed by Product at $128,000. Operations is lowest at $67,000. "
            'Analysis excluded 15% of rows due to missing salary data."\n'
            "  Confidence: medium"
        ),
    },
    {
        "role": "assistant",
        "content": """{
  "score": 8.0,
  "issues": [
    "Median salary not mentioned — useful for detecting skew in small departments"
  ],
  "suggestions": [
    "Add median salary alongside mean to give a more complete picture",
    "Note the employee count per department so the reader knows if small departments skew the average"
  ]
}""",
    },
]
