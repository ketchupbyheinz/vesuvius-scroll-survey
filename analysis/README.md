# analysis/

The scripts that turned the run into the tables in the main README. They were run on the survey machine and use its
paths (`/home/claude/vesuvius/...`, `/mnt/user-data/...`) and folder layout (`tools/`, `survey/cards/`,
`survey/grades.json`); adjust them before re-running.

| file | what |
|---|---|
| `aggregate.py` | cards + sheet grades + side/curvature/crack-check values -> `results/survey.json/.csv` |
| `face_side.py`, `curv_side.py`, `dark_all.py` | per-patch side (scroll centre), curvature cross-check, crack check |
| `four_model.py` | model-pair inner/outer agreement ratios -> `results/leads/model_pairs_*.json` |
| `deep_view.py`, `deep_overlay.py` | depth-sweep profile and ensemble overlays |
| `neighbour_seeds.py`, `lateral_seeds.py`, `make_seeds4.py` | follow-up and survey seed generation |
| `leads_table.py`, `fill_readme.py`, `README.template.md`, `takeaways.md` | build the main README |
| `fig_pos_neg.py` | `docs/img/pos_vs_neg.png` |
| `tile_asym.py`, `three_model.py` | early versions of the tile and model-pair checks |
| `seeds_*.json` | every seed list (`[scroll, scan, x, y, z, (volume, voxel um)]`); survey: `seeds_survey*.json`, `seeds_round2.json`; controls: `seeds_pos343.json`; lead follow-ups: `seeds_lead211.json`, `seeds_leads2..7.json` |
| `sheet_grades.json` | sheet-check grade of every trace (quick-look render) |
| `side.json`, `curvature.json`, `crack_check.json` | per-patch values merged into `results/survey.json` |

The 16 hand-judged traces behind the 0.31 sheet-check threshold were judged by eye on quick-look renders early in the
project; their list is not included here, only the crossness ranges reported in the main README.
