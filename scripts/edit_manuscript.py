"""
Edit ToxicityIndex_2026_IJDP_tracked.docx with:
1. Fix timezone correction section (3.5) - use actual metadata
2. Add diverging party temporal trends to section 4.4
3. Add VoteView ideology cross-reference to section 4.2
4. Update section 5.3 discussion with diverging trends
"""
from docx import Document

path = "/home/joneill/x_box/pipeline/manuscript/ToxicityIndex_2026_IJDP_tracked.docx"
doc = Document(path)


def replace_paragraph_text(para, new_text):
    if para.runs:
        font_name = para.runs[0].font.name
        font_size = para.runs[0].font.size
        bold = para.runs[0].font.bold
        italic = para.runs[0].font.italic
        for run in para.runs:
            run.text = ""
        para.runs[0].text = new_text
        para.runs[0].font.name = font_name
        para.runs[0].font.size = font_size
        para.runs[0].font.bold = bold
        para.runs[0].font.italic = italic
    else:
        para.text = new_text


paras = doc.paragraphs

# 1. P49: Timezone correction (Section 3.5)
old_tz = paras[49].text
assert "UTC" in old_tz or "timezone" in old_tz.lower(), f"P49 mismatch: {old_tz[:80]}"

replace_paragraph_text(paras[49], (
    "The congresstweets daily archive stores each tweet with an ISO 8601 timestamp "
    "including the UTC offset at collection time (e.g., 2017-06-22T15:08:29-04:00). "
    "All 1,677,042 timestamps carry either -04:00 (Eastern Daylight) or -05:00 (Eastern Standard), "
    "reflecting the scraping infrastructure's Eastern time zone with daylight saving transitions. "
    "We parsed the local datetime component directly from this field, avoiding any estimation of "
    "members' physical locations. For the supplementary Mike Lee archive, which provides only UTC "
    "timestamps, we converted to America/Denver (Mountain time) using Python's zoneinfo library, "
    "which applies historically accurate daylight saving rules. Night posting intensity was computed "
    "as the share of tweets posted between 00:00 and 05:59 local time. Under this metadata-driven "
    "correction, the population night posting rate was 1.5%, compared to artificially inflated rates "
    "produced by naive UTC analysis or state-averaged offset heuristics."
))
print("Fixed P49: Timezone correction")

# 2. P56: Party differences - add VoteView cross-reference
old_party = paras[56].text
assert "Republican" in old_party and "Democrat" in old_party, f"P56 mismatch: {old_party[:80]}"

replace_paragraph_text(paras[56], (
    "Republicans (n=196) scored significantly higher than Democrats (n=182) on the composite "
    "Toxicity Index (0.173 vs. 0.138, t=+5.1, p<0.001, d=+0.53). Decomposing TI by flag dimension "
    "revealed that party differences concentrate entirely in the custom-trained provocation classifiers "
    "and the genuine anger correction, not in standard toxicity or offensiveness measures. Offensive "
    "language showed no party difference (d=+0.005, p=0.96) and toxicity showed no party difference "
    "(d=-0.074, p=0.47). Cross-referencing with DW-NOMINATE ideology scores from VoteView (n=380 matched "
    "members) showed a moderate correlation between conservatism and TI (Pearson r=0.30, p<0.001), "
    "but ideology explained only 9% of TI variance, indicating that the Toxicity Index captures "
    "communication style rather than ideological position. TI by ideology quartile followed a "
    "curvilinear pattern: center-left members scored lowest (TI=0.120), followed by center-right "
    "(0.141), most liberal (0.156), and most conservative (0.208)."
))
print("Fixed P56: Party differences with VoteView data")

# 3. P62: Temporal trends - add diverging party trends
old_temporal = paras[62].text
assert "increased" in old_temporal.lower() or "slope" in old_temporal.lower(), f"P62 mismatch: {old_temporal[:80]}"

replace_paragraph_text(paras[62], (
    "Congressional TI increased over the study period (slope=+0.005/year, r=0.873, p=0.010), "
    "rising from a mean of 0.177 in 2017 to 0.199 in 2023. Disaggregating by party revealed "
    "diverging trajectories: Republican TI increased at +0.026 per year (r=0.920), while Democratic "
    "TI decreased at -0.015 per year (r=-0.923). The overall upward trend is therefore driven "
    "entirely by the Republican trajectory; Democrats became less provocative over the same window. "
    "This divergence was not visible in the aggregate trend and suggests that the parties are "
    "adopting increasingly different communication strategies over time, with Republican accounts "
    "shifting toward provocation tactics while Democratic accounts shift toward policy advocacy."
))
print("Fixed P62: Temporal trends with diverging party data")

# 4. P76: Discussion 5.3 - first paragraph about provocation increase
old_76 = paras[76].text
if "Composite TI" in old_76 or "five years" in old_76 or "provocation" in old_76:
    replace_paragraph_text(paras[76], (
        "Composite TI increased overall, but the aggregate trend conceals a striking divergence. "
        "Republican accounts became substantially more provocative (+0.026/year, r=0.920), while "
        "Democratic accounts became less provocative (-0.015/year, r=-0.923). The overall increase "
        "is driven entirely by one party. This pattern is consistent with asymmetric polarization "
        "theories (Jost et al., 2022) and with structural accounts linking online hostility to "
        "political incentives (Petersen et al., 2026). It also raises the possibility that platform "
        "design interacts differently with different communication strategies: provocation-oriented "
        "accounts may receive differential algorithmic amplification, creating a feedback loop that "
        "selectively rewards one style over another."
    ))
    print("Fixed P76: Discussion 5.3 with diverging trends")

# 5. P77: structural account paragraph
old_77 = paras[77].text
if "Petersen" in old_77 and "thirty countries" in old_77:
    replace_paragraph_text(paras[77], (
        "This temporal trajectory is consistent with structural accounts of online political hostility. "
        "Petersen et al. (2026) report that hostility levels co-vary with national economic inequality "
        "and democratic strength across thirty countries. The 2017 to 2023 US window encompasses "
        "measurable rises in both inequality and democratic backsliding indicators (Bermeo, 2016; "
        "Levitsky & Ziblatt, 2018). However, the party-specific divergence reported here goes beyond "
        "a uniform structural effect. If structural conditions alone drove the trend, both parties would "
        "track upward together. The Republican-specific acceleration and Democratic-specific deceleration "
        "suggest that strategic choices, constituency incentives, or differential platform dynamics play "
        "a role beyond macrostructural forces."
    ))
    print("Fixed P77: Structural account with party divergence")

doc.save(path)
print(f"\nSaved: {path}")
print("DONE")
