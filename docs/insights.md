# Portfolio insights

From `notebooks/02_data_analytics.ipynb` (charts and full tables there). Portfolio: 14,685 loan accounts,
10,000 customers, $601.1M exposure (original loan amount), 16.1% default rate.

Each insight follows the same pattern: **segment = share of exposure, how much higher its default rate is than the
portfolio's 16.1%**, then the "so what".

| # | Insight | So what |
|---|---|---|
| 1 | **Mortgages = 52.3% of exposure, 53% higher default rate** (24.7%). They are 9.8% of accounts but hold 65.0% of defaulted exposure; the median mortgage is 2.8x the borrower's annual income, against 0.2x for other products. | The mortgage book is the biggest lever: review mortgage underwriting first, and obtain collateral (LTV) data, which this dataset lacks, to size real losses. |
| 2 | **Loans with an instalment of 30%+ of monthly income = 32.6% of exposure, 59% higher default rate** (25.7%). Above 50% of income: 10.8% of exposure, 91% higher (30.8%). 79 loans have an instalment larger than the whole income: 183% higher (45.6%). | Introduce a hard affordability cap at origination (e.g. instalment at most 40% of income). The 79 breaches show impossible loans get through today. |
| 3 | **Borrowers earning under $50k = 12.7% of exposure, 74% higher default rate** (27.9%). Under $30k: 140% higher (38.7%). They hold 21.3% of defaulted exposure. | Income-tiered limits and pricing; verified income for applicants under $50k. |
| 4 | **Cross-cut, income x employment: unemployed borrowers under $30k = 1.6% of exposure, 175% higher default rate** (44.3%, 228 accounts). Salaried borrowers on $100k+ (11.1% of exposure) default 44% *less* than average (9.0%). Unemployed customers default more at every income level. | Require verified alternative income (pension, rental) for unemployed applicants and route them to manual underwriting. |
| 5 | **Cross-cut, region x loan type: South-region mortgages = 14.0% of exposure, 70% higher default rate** (27.3%, 381 accounts). Region on its own barely matters (15.0%–16.9%); mortgages run 21.3%–27.3% in every region. | Don't build region-level credit policy; review the South mortgage book instead. |

**Also worth knowing**
- Young borrowers default more (21–25: 22.2% vs 56+: 10.6%), but age is excluded from the model for fair-lending
  reasons, at no cost in accuracy.
- Gender (16.3% vs 16.0%), channel, loan purpose, sector, housing and city tier don't separate risk.
- Default rates are flat across origination years (14.9%–17.7%): no vintage deterioration.
- A bureau write-off history lifts the default rate less than expected (18.7% vs 15.0%); a missed payment in the first
  year is a stronger early signal (20.6% vs 12.9%).
