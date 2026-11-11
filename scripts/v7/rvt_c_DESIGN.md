# RVT-C: the application of Recovery Viability Theory redefined for the present data

Date written: 2026-11-10, before any RVT-C estimate was computed. The results of RVT-A and RVT-B (designs of 2026-10-13) were known when this was written, so RVT-C is a second, informed design and its results are reported as such. This file is not edited after estimation; deviations are logged in the report.

## Why the earlier application no longer fits the data

RVT-A and RVT-B were written when the analysis unit was the post and the only state variable tested was content.

| Element | RVT-A and RVT-B (October) | Data now |
|---|---|---|
| State | Content only (negative-partisan share, TI7) | Three coordinates per member: content (TI7), volume (event rate) and rhythm; events, with threads combined |
| Environment | Party control, annual or at four dates | Three scales: activity shared across Congress (hours), calendar (days), party control (years) |
| Recurring circumstance | Loss of party control: five in the panel, three in the corpus, most members see one or two | Each member's own general election: three cycles in the corpus (2018, 2020, 2022), about 420 members on the ballot in each, the same constraint returning to the same member on a known schedule |
| Experience | A binary: earlier time out of power, compared between members | Outcome of the member's own previous election, compared within the member |
| Constraint set | "Not observed" | The general-election window (Labor Day to election day) of a member on the ballot is a dated period in which an electoral constraint binds |
| Resources | Seniority, margin, majority, chamber, ideology as predictors of a response to party control | The margin of the previous election as the resource that should loosen or tighten the electoral constraint |
| Sudden change | Party means at four dates, content only | Same, plus volume |
| Out of sample | None | The change in party control of January 2025 (case-study period) |

## Redefined mapping

- State of member i in a period: x = (mean TI7 of non-repost posts, log event rate). Rhythm enters through the timing models of Section 4.2 and is not retested here.
- Zone: the region of x a member occupies in a reference period. A constraint that binds should move or shrink it.
- Constraint: own general-election window for members on the ballot (calendar_member_day: general_window, on_ballot_this_cycle).
- Resource: margin of the member's previous general election (h3_data.csv margin at the following transition; House members mostly).
- Experience: the member's previous passage through the same constraint and its outcome (that margin).
- Environment: party control at 2019-01-03, 2021-01-20, 2023-01-03.

## Tests (corpus, member level; outcomes TI7 and log event rate; member-clustered SE; two-sided 0.05; Holm across the two outcomes within a test)

Response of member i in cycle c: R(i,c) = state in the general-election window minus state in the 56 days before Labor Day. TI7 is post-weighted over the period (at least 10 non-repost posts in each period); log event rate is log((events + 0.5)/days in office).

- C1 Constraint binds. R of members on the ballot minus R of members in office and not on the ballot, cycle fixed effects. RVT: the zone contracts, so on-ballot members fall in volume or TI7 relative to the others. Supported if either outcome is negative with Holm p < 0.05.
- C2 Resource loosens the constraint. Among on-ballot members with a previous margin: R(i,c) on previous margin (standardized), cycle fixed effects, party, chamber. RVT: safer members contract less (positive coefficient). Supported if positive with Holm p < 0.05 on the outcome(s) that contracted in C1; if nothing contracted in C1, C2 is not interpretable and is reported descriptively.
- C3 Experience rewrites the response (within member). For members on the ballot in consecutive cycles: R(i,c) - R(i,c-1) on the margin of cycle c-1 (standardized), cycle fixed effect. Fixed landscape: zero. RVT: nonzero. Also reported: correlation of R(i,c) with R(i,c-1). Supported if the coefficient differs from zero with Holm p < 0.05.
- C4 Environment moves the whole state. For each change in party control: displacement of each coordinate (180 days after minus 180 days before, as in H3), members who lost power minus members who gained, pooled over the three changes with change fixed effects. Placebo: the same contrast one year before each change. RVT (temporal drift): nonzero for TI7 (known from H3) and tested here for volume. Supported for a coordinate if Holm p < 0.05 and the placebo contrast is not.
- C5 Sudden change in volume. H4a's step and ramp models applied to the losing party's daily mean log event count at the three change dates, with the same 50 placebo dates rule. Reported with the TI7 result.

## Out of sample

Whatever C4 finds for the corpus gives a signed prediction for January 2025, when the Democrats lost the Senate and the presidency. It can be checked on the case-study senator's two accounts (content, already reported: yearly TI7 lower in 2025 on both, the predicted direction for a member whose party gained power) and, for volume, on the 36 office accounts if C4 finds a volume response. No further use of the 2025 data is made before C4 is fixed.
