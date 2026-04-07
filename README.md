# Green Bond Premium (Greenium) Estimation

## Research Question
Do green bonds trade at a premium (lower yield) compared to conventional bonds?

## Methodology
**Language:** Python  
**Methods:** Propensity score matching, yield spread analysis

## Data
Simulated bond data calibrated to published greenium estimates

## Key Findings
Green bonds show a small but significant yield discount (~5–15 bps) vs matched conventional bonds.

## How to Run
```bash
pip install -r requirements.txt
python code/project11_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
