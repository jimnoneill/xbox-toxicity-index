# RVT-D: release of the electoral constraint, and the width of the zone

Date written: 2026-11-11, before any RVT-D estimate was computed. Known when this was written: all results of RVT-A, RVT-B and RVT-C, and the number of departing members per cycle (on the ballot and departing: 40, 24, 23; not on the ballot and departing: 48, 37, 52; on the ballot and returning: 378, 407, 393), counted while checking that the calendar table identifies them. No state or width of any group had been looked at. This file is not edited after estimation; deviations are logged in the report.

## Why

RVT-C tested the electoral constraint by asking whether the mean state of members on the ballot moved down in the general-election window. It moved up in volume. Two readings of the theory were left untested.

1. A constraint is also visible when it is removed. Members who are defeated lose the electoral constraint on a known day, election day, and stay in office for about eight more weeks.
2. A zone can narrow without its centre moving. The first test looked at the mean of each coordinate; the width of each coordinate within the member was not measured.

## Definitions (congressional corpus; members of the two major parties; cycles 2018, 2020, 2022)

- Periods per cycle: pre = the 56 days before Labor Day; window = Labor Day through election day; post = the day after election day through 2 January.
- Coverage: in office on at least 90 percent of the days of the window, at least 50 days of pre and, for tests that use it, at least 50 days of post.
- State, as in RVT-C: mean TI7 over non-repost posts (at least 10 posts in the period) and log event rate, ln((events + 0.5)/days in office).
- Width of TI7: standard deviation of TI7 across the member's non-repost posts in the period (at least 20 posts in the period). A post-level standard deviation does not shrink mechanically when a member posts more.
- Width of volume: excess dispersion of the daily event counts in the period, (variance - mean)/mean squared, which is zero for a constant daily rate whatever its level and grows with the variability of the daily rate (at least 20 events in the period).
- Groups, from the calendar table: released = on the general-election ballot and departing (lame_duck = 1 in post); returned = on the ballot and not departing; departing without the ballot = not on the ballot and departing (retiring, defeated in a primary, or seeking another office); continuing = not on the ballot and not departing.

## Tests (ordinary least squares, standard errors clustered by member, two-sided 0.05, Holm across the two coordinates within a test)

- D1 Release. L = state in post minus state in the window. L regressed on group (reference: returned), party, chamber and cycle indicators. Contrast a: released minus returned. Contrast b: released minus departing without the ballot, which separates release on election day from the winding down of an office that every departing member shares. RVT: removing a binding constraint moves the state, so contrast a is nonzero. Supported for a coordinate if contrast a has Holm p < 0.05 and contrast b has the same sign with p < 0.05. If only contrast a holds, the result is reported as release not separable from leaving office. Check reported with it: the same contrast a for R = state in the window minus state in pre, which shows whether the groups had already diverged before election day.
- D2 Width under the constraint. W = width in the window minus width in pre, winsorized at the 1st and 99th percentiles of the analysis sample because excess dispersion is heavy tailed. W regressed on being on the ballot, with cycle indicators. RVT: a binding constraint narrows the zone, so the coefficient is negative. Supported if negative with Holm p < 0.05 in either coordinate. A positive significant coefficient is reported as the opposite of the prediction.
- D3 Width at release. WL = width in post minus width in the window, winsorized in the same way, in the model of D1. RVT: release widens the zone, so contrast a is positive. Same decision rule as D1.

## Caveat fixed in advance

A fixed response to incentives also predicts that defeated members change their posting once the election is over. D1 and D3 can therefore show that the electoral constraint acts on the state; they cannot by themselves show that the set of sustainable states changed. The direction of a constraint's effect on activity (suppressing or forcing it) is not assumed in D1, which is two-sided.
