"""Fixed evaluation scenarios for qualitative and keyword-based agent testing."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any
from typing import Any


@dataclass
class TestScenario:
    name: str
    description: str
    csv_data: str
    question: str
    expected_outcome: str
    expected_keywords: list[str]


TEST_SCENARIOS: list[TestScenario] = [
    TestScenario(
        name="regional_sales",
        description="Rank categorical regions by a numeric revenue column.",
        csv_data=textwrap.dedent(
            """\
            region,revenue,product
            North,15000,Widget
            South,22000,Widget
            East,9000,Gadget
            West,31000,Gadget
            """
        ).strip(),
        question="What are the top performing regions by revenue?",
        expected_outcome=(
            "A good answer ranks regions by total revenue, names West as the top "
            "region (~$31,000), and cites specific revenue figures for other regions."
        ),
        expected_keywords=["West", "revenue", "region", "31000"],
    ),
    TestScenario(
        name="department_salary",
        description="Compute grouped averages when some salary values are missing.",
        csv_data=textwrap.dedent(
            """\
            department,salary,tenure_years
            Engineering,142000,5
            Engineering,138000,2
            Product,128000,4
            Product,125000,6
            Operations,67000,8
            Operations,,4
            Sales,95000,3
            """
        ).strip(),
        question="What is the average salary by department?",
        expected_outcome=(
            "A good answer groups by department, handles or notes missing salaries, "
            "and reports Engineering as the highest average with approximate figures."
        ),
        expected_keywords=["Engineering", "salary", "department", "average"],
    ),
    TestScenario(
        name="product_totals",
        description="Aggregate revenue across rows sharing the same product label.",
        csv_data=textwrap.dedent(
            """\
            product,revenue,units
            Widget,45000,1200
            Gadget,52000,800
            Widget,38000,900
            Gadget,41000,650
            """
        ).strip(),
        question="What is the total revenue for each product?",
        expected_outcome=(
            "A good answer sums revenue by product, reports Widget and Gadget totals, "
            "and identifies which product earned more overall."
        ),
        expected_keywords=["Widget", "Gadget", "revenue", "total"],
    ),
    TestScenario(
        name="returns_with_nulls",
        description="Analyze return rates when units_sold contains missing values.",
        csv_data=textwrap.dedent(
            """\
            product,returns,units_sold
            Widget,12,500
            Gadget,8,
            Widget,5,300
            Gadget,15,200
            Widget,3,400
            """
        ).strip(),
        question="Which product has the higher return rate, and how did you handle missing data?",
        expected_outcome=(
            "A good answer addresses missing units_sold, computes or compares return rates "
            "for Widget and Gadget, and states any data-quality caveats."
        ),
        expected_keywords=["Widget", "Gadget", "return", "missing"],
    ),
    TestScenario(
        name="monthly_sales_peak",
        description="Identify the peak month from a simple monthly sales series.",
        csv_data=textwrap.dedent(
            """\
            month,sales
            Jan,10000
            Feb,12000
            Mar,15000
            Apr,11000
            May,13500
            Jun,9000
            """
        ).strip(),
        question="Which month had the highest sales?",
        expected_outcome=(
            "A good answer identifies March as the peak month with $15,000 in sales "
            "and may briefly compare it to other months."
        ),
        expected_keywords=["Mar", "15000", "highest", "sales"],
    ),
]
