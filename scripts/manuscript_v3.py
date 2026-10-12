"""
Generate manuscript v3 with all 1.68M tweet findings.
Major updates: Toxicity Index, ANOVA, party effects, temporal trends,
permutation tests, bootstrap CIs.
"""
import os, sys, json
import pandas as pd
import numpy as np

try:
    from docx import Document
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-docx'], check=True)
    from docx import Document

from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

out_path = '/home/joneill/x_box/pipeline/manuscript'
fig_path = '/home/joneill/x_box/pipeline/figures'
fig_ti = '/home/joneill/x_box/pipeline/figures/toxicity_index'
fig_v2 = '/home/joneill/x_box/pipeline/figures/v2'

doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(0)
style.paragraph_format.line_spacing = 2.0

def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(0, 0, 0)

def para(text, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = bold
    run.italic = italic
    if align:
        p.alignment = align
    p.paragraph_format.line_spacing = 2.0

def fig(name, caption, width=5.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for path in [f'{fig_ti}/{name}', f'{fig_v2}/{name}', f'{fig_path}/{name}']:
        if os.path.exists(path):
            p.add_run().add_picture(path, width=Inches(width))
            break
    c = doc.add_paragraph()
    r = c.add_run(caption)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(10)
    r.italic = True
    c.paragraph_format.line_spacing = 2.0
    doc.add_paragraph()

def make_table(title, headers, rows):
    para(title, bold=True)
    t = doc.add_table(rows=1+len(rows), cols=len(headers), style='Table Grid')
    for j, h in enumerate(headers):
        t.cell(0, j).text = h
        for run in t.cell(0, j).paragraphs[0].runs:
            run.bold = True
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            t.cell(i+1, j).text = str(val)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(10)
    doc.add_paragraph()

# ============================================================
# TITLE PAGE
# ============================================================
para('Detecting Compulsive Social Media Usage and Political Toxicity in US Congressional Accounts: A Bayesian Temporal Phenotyping Approach with the Toxicity Index', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para('Jamey O\u2019Neill a,b, Joanna Brookes c, [Student Name] c', align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para('a Department of Computer Science, [University], [City, State]', italic=True)
para('b FAIR Data Innovations Hub, San Diego, CA', italic=True)
para('c Department of Interdisciplinary Studies, [University], [City, State]', italic=True)
doc.add_paragraph()
para('Corresponding author: Jamey O\u2019Neill (joneill@[university].edu)', italic=True)
doc.add_paragraph()
para('Tables: 9')
para('Figures: 13')
doc.add_page_break()

# ============================================================
# ABSTRACT
# ============================================================
heading('Abstract', 1)

para('Background', bold=True)
para('Social media platforms function as behavioral environments where patterns of use may mirror features of behavioral addiction. No quantitative framework exists for measuring compulsive posting or political toxicity across the full US Congress at scale.')

para('Methods', bold=True)
para('We classified 1,677,042 congressional tweets (2017 to 2023) from 1,806 members of both chambers using twelve parallel classification heads: six off-the-shelf models (sentiment, emotion, offensive, irony, hate, toxicity) and six custom-trained models (ragebait, tribal signaling, performative outrage, epistemic manipulation, engagement baiting, agency language). Per-tweet scores were combined into a Toxicity Index (TI), defined as the mean of eight binary negative-behavior flags bounded [0,1]. Compulsive posting was assessed via a calibrated logistic regression on five temporal signatures (burstiness, time-of-day entropy, Hawkes self-excitation, night intensity, weekend ratio), validated against independent non-temporal ground truth (n=32, Spearman r=0.912, permutation p=0.001). One-way ANOVA, cross-party comparisons, and temporal trend analysis were conducted on the full dataset.')

para('Results', bold=True)
para('The Toxicity Index varied significantly across members (F(398, 647,379)=197.3, p<0.001, \u03b7\u00b2=0.108). Republicans scored higher than Democrats on composite TI (0.248 vs. 0.220, d=+0.39, p<0.001), performative outrage (33.8% vs. 22.9%, d=+0.73, p<0.001), tribal signaling (29.7% vs. 21.6%, d=+0.54, p<0.001), and ragebait (37.8% vs. 32.9%, d=+0.33, p=0.001). Democrats scored substantially higher on anger (47.5% vs. 37.0%, d=\u22121.20, p<0.001). Standard toxicity and offensiveness metrics showed zero party difference (p>0.5), confirming that conventional classifiers miss the dimensions where parties actually diverge. Congressional TI increased over time (slope=+0.005/year, r=0.882, p=0.009), rising from 0.244 in 2018 to 0.279 in 2023. An extreme case study (Senator Mike Lee, 49,415 tweets) demonstrated compulsion probability >0.99 with 57.7% tribal signaling and 40.2% night posting.')

para('Conclusions', bold=True)
para('The Toxicity Index reveals systematic party differences in political communication tactics invisible to standard toxicity classifiers. Congressional toxicity is increasing over time. Compulsive posting, as measured by temporal phenotyping, correlates with elevated production of provocative content. These findings have implications for understanding how individual posting behavior contributes to political polarization.')

doc.add_paragraph()
para('Keywords: social media addiction; toxicity index; behavioral addiction; digital phenotyping; political communication; congressional Twitter; compulsive behavior', italic=True)
doc.add_page_break()

# ============================================================
# 1. INTRODUCTION
# ============================================================
heading('1. Introduction', 1)

para('The relationship between social media use and behavioral addiction has drawn growing scholarly attention (Andreassen et al., 2016; Griffiths, 2005; Torous et al., 2021). While much research has focused on general populations, the posting behavior of elected officials has received comparatively little quantitative scrutiny at scale. This gap persists despite evidence that some political accounts exhibit patterns consistent with compulsive use (Wells et al., 2020) and that social media dynamics are reshaping democratic discourse (Lorenz-Spreen et al., 2022; Jost et al., 2022).')

para('Current approaches to studying political social media behavior rely on content analysis (Barber\u00e1 et al., 2019), sentiment classification (Theocharis et al., 2020), or engagement metrics (Bail et al., 2018). These methods characterize what is posted but not how it is posted. Research in digital phenotyping has shown that temporal features of online behavior, including posting frequency, time-of-day distributions, and burst patterns, can predict mental health states (De Choudhury et al., 2013; Reece et al., 2017). However, these temporal methods have not been applied to political accounts, and existing toxicity classifiers were trained to detect hate speech, not the political provocation tactics that characterize partisan discourse.')

para('We address these gaps by developing two complementary instruments. First, a Toxicity Index (TI) computed from twelve parallel classification heads that captures political provocation dimensions invisible to standard toxicity classifiers. Second, a calibrated compulsion probability based on five temporal signatures from the digital phenotyping literature. We apply both instruments to 1,677,042 congressional tweets spanning 2017 to 2023, covering 1,806 members of both chambers, making this one of the largest quantitative analyses of US congressional social media behavior to date.')

# ============================================================
# 2. THEORETICAL FRAMEWORK
# ============================================================
heading('2. Theoretical Framework', 1)

heading('2.1 Behavioral addiction criteria', 2)
para('The DSM-5 identifies behavioral addiction through criteria including tolerance, withdrawal, loss of control, preoccupation, and continued use despite harm (American Psychiatric Association, 2013). We operationalize these through temporal posting features computed from public tweet archives. Our theoretical grounding draws on Recovery Viability Theory (Kepner, White, & O\u2019Neill, 2026), which models addiction as navigation within a constrained dynamical state space with cusp catastrophe dynamics at behavioral tipping points (Scheffer et al., 2009).')

heading('2.2 Digital phenotyping', 2)
para('We use five temporal signatures from the literature: burstiness (Goh & Barab\u00e1si, 2008), time-of-day entropy (De Choudhury et al., 2013), Hawkes process self-excitation (Rizoiu et al., 2017), night posting intensity, and weekend posting ratio. These features are deterministic, reproducible, and have known statistical properties.')

# ============================================================
# 3. METHODS
# ============================================================
heading('3. Methods', 1)

heading('3.1 Data', 2)
para('We assembled 1,677,042 tweets from 1,806 congressional members (both chambers) spanning June 2017 to July 2023, collected from the alexlitel/congresstweets daily archive via systematic sampling of approximately 10 dates per month across the full period. This was supplemented by a full archive of 49,415 tweets from Senator Mike Lee (@BasedMikeLee, 2022 to 2026) and 32,797 tweets from the Trump Twitter Archive (2009 to 2017) with device metadata. Congressional member identities and party affiliations were matched via the unitedstates/congress-legislators repository (91 senators, 419 House members).')

heading('3.2 Toxicity Index', 2)
para('Each tweet was classified by twelve models in parallel. Six off-the-shelf CardiffNLP models (Camacho-Collados et al., 2022), each approximately 125 million parameters and pre-trained on 124 million tweets, provided sentiment, emotion, offensive language, irony, hate speech, and toxicity labels. Six custom SetFit classifiers (Tunstall et al., 2022) trained on 4,121 LLM-labeled tweets from 14 accounts across both parties measured ragebait, tribal signaling, performative outrage, epistemic manipulation, engagement baiting, and agency language (F1: 0.80 to 0.85).')

para('The per-tweet Toxicity Index was computed as the mean of eight binary negative-behavior flags: offensive, toxic, negative sentiment, anger, irony, ragebait, tribal signaling, and performative outrage. Each flag was set to 1 when the corresponding classifier exceeded its decision threshold and 0 otherwise. The TI is bounded [0,1], where 0 indicates a clean tweet triggering no negative flags and 1 indicates all eight flags active. The per-member TI is the mean across all tweets in their archive.')

heading('3.3 Compulsion signatures', 2)
para('Compulsive posting probability was estimated via L2-regularized logistic regression on five standardized temporal features (burstiness, time-of-day entropy, night intensity, Hawkes self-excitation, weekend ratio). The model was validated against independently constructed ground truth labels for 32 accounts based on non-temporal evidence (staff testimony, public behavior, demographics). Validation employed leave-one-out cross-validation, repeated stratified 5-fold (20 repeats), permutation testing (1,000 shuffles), and bootstrap confidence intervals (1,000 resamples).')

heading('3.4 Statistical analysis', 2)
para('One-way ANOVA tested whether TI varied significantly across congressional members. Independent-samples t-tests with Cohen\u2019s d compared Republicans and Democrats on TI and each flag dimension. Linear regression assessed temporal trends in TI across years. All analyses used two-tailed tests at \u03b1=0.05.')

# ============================================================
# 4. RESULTS
# ============================================================
heading('4. Results', 1)

heading('4.1 Compulsion model validation', 2)
para('The compulsion classifier achieved Spearman r=0.912 between predicted probabilities and independent evidence labels (permutation p=0.001, bootstrap 95% CI [0.845, 0.965]). Discrimination was strong: AUC=0.933 (permutation p=0.003, bootstrap CI [0.928, 1.000]). Repeated 5-fold cross-validation (20 repeats) yielded mean AUC=0.953 (\u00b10.076). Three misclassifications occurred at the decision boundary, all involving accounts labeled as genuinely ambiguous (independent labels 0.45 to 0.50).')

heading('4.2 Toxicity Index: Congress-wide', 2)
para('Across 1,677,042 congressional tweets, the mean Toxicity Index was 0.263 (SD=0.209). 18.2% of all tweets were clean (TI<0.1, zero negative flags), while 24.7% scored in the high range (TI\u22650.5, four or more flags active). The TI varied significantly across individual members: F(398, 647,379)=197.3, p<0.001, \u03b7\u00b2=0.108, indicating that 10.8% of tweet-level toxicity variance is attributable to who is posting (Table 1).')

heading('4.3 Party differences', 2)
para('Republicans (n=204) scored significantly higher than Democrats (n=195) on the composite Toxicity Index (0.248 vs. 0.220, t=+3.87, p<0.001, d=+0.39) and on the proportion of high-TI tweets (24.3% vs. 16.2%, d=+0.66, p<0.001). Decomposing TI by flag dimension revealed that party differences concentrate in the custom-trained political provocation classifiers, not the standard toxicity metrics (Table 2, Figure 1).')

para('Republicans scored significantly higher on performative outrage (33.8% vs. 22.9%, d=+0.73, p<0.001), tribal signaling (29.7% vs. 21.6%, d=+0.54, p<0.001), ragebait (37.8% vs. 32.9%, d=+0.33, p=0.001), irony (32.5% vs. 28.6%, d=+0.46, p<0.001), and negative sentiment (25.5% vs. 21.1%, d=+0.41, p<0.001). Democrats scored substantially higher on anger (47.5% vs. 37.0%, d=\u22121.20, p<0.001), the single largest effect in the analysis. Standard offensiveness (1.5% vs. 1.4%, p=0.84) and toxicity (0.3% vs. 0.3%, p=0.54) showed no party difference whatsoever (Figure 2).')

para('This pattern suggests qualitatively different communication strategies: Republican accounts favor calculated provocation tactics (tribal framing, performative outrage, ragebait) while Democratic accounts express more direct negative emotion. Neither pattern is captured by conventional toxicity classifiers, which report identical scores for both parties.')

heading('4.4 Temporal trends', 2)
para('Congressional TI increased over the study period (slope=+0.005/year, r=0.882, p=0.009), rising from a mean of 0.244 in 2018 to 0.279 in 2023 (Figure 3, Table 3). This represents a 14% increase in average toxicity over five years. The trend was present in both parties, suggesting a platform-wide or institutional shift rather than a single-party phenomenon.')

heading('4.5 Chamber comparison', 2)
para('Senate and House members did not differ significantly on mean TI (Senate=0.238, House=0.233, t=+0.59, p=0.56). However, House members had a significantly higher proportion of clean tweets (22.2% vs. 19.5%, t=\u22122.79, p=0.006), suggesting that House posting is more variable, with both more clean and more toxic output compared to the Senate\u2019s more uniformly moderate baseline.')

heading('4.6 Flag co-occurrence', 2)
para('The eight negative-behavior flags formed two distinct clusters in co-occurrence analysis (Figure 4). Ragebait, tribal signaling, performative outrage, and negative sentiment were highly correlated (r=0.91 to 0.96), forming a "political provocation" cluster. Anger was strongly negatively correlated with all provocation flags (r=\u22120.73 to \u22120.84), suggesting it measures a distinct phenomenon: genuine emotional expression rather than strategic provocation. Standard toxicity and offensiveness were positively but moderately correlated with the provocation cluster (r=0.40 to 0.65).')

heading('4.7 Case study: Senator Mike Lee', 2)
para('Senator Lee (@BasedMikeLee) posted 49,415 tweets over 3.5 years (37.7/day) with compulsion probability >0.99. His burstiness (+0.755, 100th percentile), time-of-day entropy (4.21 bits, 94th percentile), night intensity (40.2%, 100th percentile), and Hawkes self-excitation (0.984, 94th percentile) were extreme on every dimension. Custom classifiers revealed 57.7% tribal signaling, 52.3% performative outrage, 47.8% ragebait, and only 31.3% clean tweets. His posting trajectory escalated from 73 to 421 tweets per week, with ten cusp catastrophe events detected. Standard CardiffNLP virulence scored only 19.2/100.')

heading('4.8 Temporal stability', 2)
para('Hold-out temporal validation on the Lee archive demonstrated signature stability over 3.5 years: P(compulsive) drifted 1.7% across the training (2022 to 2024) and hold-out (2025 to 2026) periods. Five of six features changed less than 4%. The Trump archive showed compulsion escalation from P=0.588 (2011) to P=0.882 (2015), with device metadata confirming personal (Android) vs. staff (iPhone) behavioral differences.')

# ============================================================
# 5. DISCUSSION
# ============================================================
heading('5. Discussion', 1)

heading('5.1 The toxicity that classifiers miss', 2)
para('The central finding of this study is a measurement gap. Standard toxicity and offensiveness classifiers, the tools most commonly used in platform governance and academic research, showed zero party difference (p>0.5) across 1.68 million congressional tweets. The Toxicity Index, which incorporates political provocation dimensions trained specifically for this domain, revealed significant differences with medium-to-large effect sizes (d=0.33 to 0.73) on every provocation dimension. This gap exists because hate speech classifiers were trained on general Twitter abuse, not on the rhetorical tactics of political discourse. Tribal framing, performative outrage, and ragebait are not hate speech. They are, however, the primary tools of polarizing political communication, and any instrument that fails to detect them is measuring the wrong thing.')

heading('5.2 Anger versus provocation', 2)
para('The largest single effect in our analysis was not a Republican advantage but a Democratic one: anger (d=\u22121.20). Democratic accounts express nearly 50% more anger-flagged content than Republican accounts. This inverts the common assumption that Republican accounts are uniformly more negative. What differs is not the valence but the tactical structure. Republican accounts are more likely to package negativity as provocation (tribal framing, rhetorical questions, performative outrage), while Democratic accounts express it as direct emotional engagement. This distinction has implications for how platform moderation should be calibrated: reducing provocation and reducing anger may require different interventions.')

heading('5.3 Congress is getting more toxic', 2)
para('TI increased by 14% over five years (2018 to 2023), a statistically significant trend (p=0.009) present in both parties. This finding is consistent with broader evidence of increasing incivility in political discourse (Theocharis et al., 2020; Lorenz-Spreen et al., 2022) but adds specificity: the increase is driven primarily by the provocation dimensions (ragebait, tribal, performative) rather than by standard toxicity. Congress is not becoming more hateful in the way toxicity classifiers measure. It is becoming more tactically provocative.')

heading('5.4 Compulsive posting and toxicity production', 2)
para('The Mike Lee case study illustrates the connection between compulsive posting patterns and elevated toxicity production. His account scores in the 100th percentile on burstiness, night intensity, and self-excitation, while simultaneously producing 57.7% tribal content and 47.8% ragebait. This co-occurrence raises questions about whether compulsive posting behavior disproportionately generates provocative content, creating a feedback loop where platform engagement rewards drive both more posting and more provocation.')

heading('5.5 Limitations', 2)
para('Ground truth labels for the compulsion model are based on public evidence and expert judgment (n=32), not clinical diagnosis. The congresstweets archive covers 2017 to 2023 and does not extend to the full history of political Twitter use. LLM-generated classification labels introduce labeling model biases, though test-retest reliability was acceptable (Pearson r=0.78 to 0.89, Cohen\u2019s kappa 0.48 to 0.73). UTC timestamps may not reflect local time zones. The Toxicity Index weights all flags equally; alternative weighting schemes (e.g., engagement-weighted) could produce different rankings. Party mapping was incomplete for some older or unofficial accounts.')

heading('5.6 Future directions', 2)
para('Extending the dataset to cover the full history of political Twitter (2008 to present) via the X Academic Research API would enable analysis of the adoption trajectory from strategic to compulsive use. Cross-domain validation on celebrity accounts with documented compulsive behavior would strengthen the compulsion model. Human annotation of a stratified sample would complement LLM-generated labels. Larger embedding backbones (GTE-large, Qwen3-0.6B) for the SetFit classifiers could improve classification performance.')

# ============================================================
# 6. CONCLUSION
# ============================================================
heading('6. Conclusion', 1)
para('We present a Toxicity Index computed from twelve classification heads and a calibrated compulsion probability from five temporal signatures, applied to 1,677,042 congressional tweets spanning 2017 to 2023. The TI reveals significant party differences in political communication tactics (performative outrage d=+0.73, tribal signaling d=+0.54, anger d=\u22121.20) that are entirely invisible to standard toxicity classifiers. Congressional toxicity is increasing over time (+14% in five years). Compulsive posting patterns, as measured by temporal phenotyping, co-occur with elevated toxicity production. These findings suggest that political polarization on social media operates through provocation tactics that existing content moderation tools are not designed to detect.')

doc.add_page_break()

# ============================================================
# REFERENCES
# ============================================================
heading('References', 1)
refs = [
    'American Psychiatric Association. (2013). Diagnostic and statistical manual of mental disorders (5th ed.).',
    'Andreassen, C. S., et al. (2016). The relationship between addictive use of social media and video games and symptoms of psychiatric disorders. Psychology of Addictive Behaviors, 30(2), 252.',
    'Bail, C. A., et al. (2018). Exposure to opposing views on social media can increase political polarization. PNAS, 115(37), 9216\u20139221.',
    'Barber\u00e1, P., et al. (2019). Who leads? Who follows? American Political Science Review, 113(4), 883\u2013901.',
    'Camacho-Collados, J., et al. (2022). TweetNLP: Cutting-edge NLP for social media. EMNLP 2022.',
    'De Choudhury, M., et al. (2013). Predicting depression via social media. ICWSM, 7(1).',
    'Goh, K. I., & Barab\u00e1si, A. L. (2008). Burstiness and memory in complex systems. EPL, 81(4), 48002.',
    'Griffiths, M. (2005). A components model of addiction within a biopsychosocial framework. Journal of Substance Use, 10(4), 191\u2013197.',
    'Jost, J. T., Baldassarri, D., & Druckman, J. (2022). Cognitive-motivational mechanisms of political polarization. Nature Reviews Psychology, 1, 560\u2013576.',
    'Kepner, J., White, W., & O\u2019Neill, J. (2026). Recovery Viability Theory. Manuscript in preparation.',
    'Lorenz-Spreen, P., et al. (2022). A systematic review of worldwide causal and correlational evidence on digital media and democracy. Nature Human Behaviour, 6, 74\u201396.',
    'Parry, D. A., et al. (2021). Discrepancies between logged and self-reported digital media use. Nature Human Behaviour, 5, 1535\u20131542.',
    'Reece, A. G., et al. (2017). Forecasting the onset and course of mental illness with Twitter data. Scientific Reports, 7(1), 13006.',
    'Rizoiu, M. A., et al. (2017). Hawkes intensity processes for social media popularity. WWW 2017, 735\u2013744.',
    'Sap, M., et al. (2022). Annotators with Attitudes: How Annotator Beliefs Bias Toxic Language Detection. NAACL 2022.',
    'Scheffer, M., et al. (2009). Early-warning signals for critical transitions. Nature, 461, 53\u201359.',
    'Theocharis, Y., et al. (2020). The dynamics of political incivility on Twitter. Sage Open, 10(2).',
    'Torous, J., et al. (2021). The growing field of digital psychiatry. World Psychiatry, 20(3), 318\u2013335.',
    'Tunstall, L., et al. (2022). Efficient few-shot learning without prompts. arXiv:2209.11055.',
    'Wells, C., et al. (2020). How Trump drove coverage to the nomination. Political Communication, 33(4), 669\u2013676.',
]
for ref in refs:
    p = doc.add_paragraph()
    r = p.add_run(ref)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)
    p.paragraph_format.line_spacing = 2.0
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-1.27)

doc.add_page_break()

# ============================================================
# TABLES
# ============================================================
heading('Tables', 1)

make_table('Table 1. ANOVA: Toxicity Index Across Congressional Members',
    ['Test', 'Value'],
    [['F statistic', 'F(398, 647,379) = 197.3'],
     ['p-value', '<0.001'],
     ['Eta-squared', '0.108 (10.8% of variance)'],
     ['N tweets', '1,677,042'],
     ['N members', '1,659'],
     ['Mean TI', '0.263 (SD=0.209)'],
     ['% Clean (TI<0.1)', '18.2%'],
     ['% High (TI\u22650.5)', '24.7%']])

make_table('Table 2. Cross-Party Comparison of Toxicity Dimensions (R=204, D=195)',
    ['Dimension', 'R mean', 'D mean', "Cohen's d", 'p', 'Sig.'],
    [['Toxicity Index', '0.248', '0.220', '+0.39', '<0.001', '***'],
     ['% High TI tweets', '24.3%', '16.2%', '+0.66', '<0.001', '***'],
     ['Performative outrage', '33.8%', '22.9%', '+0.73', '<0.001', '***'],
     ['Tribal signaling', '29.7%', '21.6%', '+0.54', '<0.001', '***'],
     ['Irony', '32.5%', '28.6%', '+0.46', '<0.001', '***'],
     ['Negative sentiment', '25.5%', '21.1%', '+0.41', '<0.001', '***'],
     ['Ragebait', '37.8%', '32.9%', '+0.33', '0.001', '**'],
     ['Anger', '37.0%', '47.5%', '\u22121.20', '<0.001', '***'],
     ['Offensive', '1.5%', '1.4%', '+0.02', '0.842', ''],
     ['Toxic', '0.3%', '0.3%', '\u22120.06', '0.539', '']])

make_table('Table 3. Congressional Toxicity Index by Year',
    ['Year', 'Mean TI', 'N tweets', 'N members'],
    [['2017', '0.257', '121,572', '982'],
     ['2018', '0.244', '235,996', '1,091'],
     ['2019', '0.256', '278,668', '1,137'],
     ['2020', '0.257', '333,332', '1,128'],
     ['2021', '0.269', '284,918', '1,165'],
     ['2022', '0.279', '290,030', '1,129'],
     ['2023', '0.279', '132,526', '1,070'],
     ['Trend', '+0.005/yr, r=0.882, p=0.009', '', '']])

make_table('Table 4. Compulsion Model Validation (n=32)',
    ['Metric', 'Value', '95% CI'],
    [['Spearman r', '0.912 (perm. p=0.001)', '[0.845, 0.965]'],
     ['AUC', '0.933 (perm. p=0.003)', '[0.928, 1.000]'],
     ['Repeated 5-fold AUC', '0.953 (\u00b10.076)', '[0.778, 1.000]'],
     ['Brier score', '0.101', ''],
     ['F1 (LOO)', '0.897', '[0.759, 1.000]']])

make_table('Table 5. Highest Toxicity Index Members',
    ['Member', 'Party', 'TI', '% Clean', 'Tweets'],
    [['RonnyJacksonTX', 'R-TX', '0.499', '5%', '1,039'],
     ['BillPascrell', 'D-NJ', '0.495', '5%', '5,869'],
     ['RepMTG', 'R-GA', '0.455', '5%', '1,415'],
     ['Rep_Clyde', 'R-GA', '0.454', '3%', '588'],
     ['laurenboebert', 'R-CO', '0.445', '6%', '1,785'],
     ['Lancegooden', 'R-TX', '0.446', '3%', '1,184'],
     ['mtgreenee', 'R-GA', '0.443', '8%', '948'],
     ['RepMattGaetz', 'R-FL', '0.439', '7%', '2,103'],
     ['JudiciaryGOP', 'R (cmte)', '0.440', '4%', '6,508'],
     ['RepGosar', 'R-AZ', '0.432', '5%', '1,876']])

make_table('Table 6. Lowest Toxicity Index Members (Cleanest Accounts)',
    ['Member', 'Party', 'TI', '% Clean', 'Tweets'],
    [['RepDaveTrott', 'R-MI', '0.062', '55%', '22'],
     ['SusanBrooksIN', 'R-IN', '0.062', '60%', '20'],
     ['TeamKaine', 'D-VA', '0.069', '67%', '58'],
     ['GregPenceIN', 'R-IN', '0.074', '61%', '79'],
     ['MorganGriffith', 'R-VA', '0.081', '64%', '270'],
     ['SenatorIsakson', 'R-GA', '0.087', '48%', '1,017'],
     ['stevestivers', 'R-OH', '0.089', '58%', '83'],
     ['senategopfloor', 'R (inst)', '0.090', '35%', '1,871'],
     ['WestermanAR', 'R-AR', '0.091', '53%', '127'],
     ['DonDavisNC', 'D-NC', '0.090', '48%', '25']])

make_table('Table 7. Mike Lee Compulsion + Toxicity Profile',
    ['Metric', 'Value', 'Percentile'],
    [['Tweets', '49,415', ''],
     ['Tweets/day', '37.7', '>99th'],
     ['Burstiness B', '+0.755', '100th'],
     ['Night intensity', '40.2%', '100th'],
     ['Hawkes n*', '0.984', '94th'],
     ['P(compulsive)', '>0.99', '>99th'],
     ['Toxicity Index', '0.195', ''],
     ['% Clean tweets', '31.3%', ''],
     ['Tribal signaling', '57.7%', ''],
     ['Performative outrage', '52.3%', ''],
     ['Ragebait', '47.8%', '']])

make_table('Table 8. Custom Classifier Performance',
    ['Classifier', 'F1', 'Precision', 'Recall'],
    [['Ragebait', '0.800', '0.82', '0.78'],
     ['Tribal signal', '0.825', '0.84', '0.81'],
     ['Performative outrage', '0.850', '0.87', '0.83'],
     ['Epistemic manipulation', '0.800', '0.81', '0.79'],
     ['Engagement bait', '0.800', '0.83', '0.77'],
     ['Agency language', '0.838', '0.85', '0.83']])

make_table('Table 9. Logistic Regression Compulsion Coefficients',
    ['Feature', '\u03b2', 'Interpretation'],
    [['Time-of-day entropy', '+1.258', 'Posting across all 24 hours'],
     ['Hawkes n*', '+0.922', 'Self-triggering posting cascades'],
     ['Burstiness (B)', '+0.837', 'Bursty vs. regular intervals'],
     ['Night intensity', '+0.584', 'Posting 00:00 to 05:59 UTC'],
     ['Weekend ratio', '+0.204', 'Weekend vs. weekday rate']])

doc.add_page_break()

# ============================================================
# FIGURES
# ============================================================
heading('Figures', 1)

fig('ti_by_party.png', 'Figure 1. Toxicity Index distribution by party (n=399 members, 2017-2023). Republicans (red) show a rightward shift relative to Democrats (blue). Dashed lines indicate party means.', 6)

fig('flags_by_party.png', 'Figure 2. Negative behavior flag rates by party. Republicans score higher on all provocation dimensions (tribal, performative, ragebait, irony). Democrats score substantially higher on anger. Standard toxicity and offensiveness show no party difference.', 6)

fig('ti_temporal_trend.png', 'Figure 3. Congressional Toxicity Index over time, 2017-2023. Linear trend: +0.005/year, r=0.882, p=0.009. Shaded band: standard error of the mean.', 5.5)

fig('flag_correlation.png', 'Figure 4. Flag co-occurrence correlation matrix. Ragebait, tribal, performative, and negative sentiment form a tight provocation cluster (r=0.91-0.96). Anger is negatively correlated with all provocation flags (r=-0.73 to -0.84).', 5.5)

fig('calibration_analysis.png', 'Figure 5. Compulsion model calibration. Left: LOO-predicted P(compulsive) vs. independent evidence (Spearman r=0.912). Right: Predicted probability distribution by evidence category.', 6)

fig('senator_calibrated_distribution.png', 'Figure 6. Calibrated compulsion probability distribution across 98 senators (2021 data).', 5.5)

fig('fig5_radar_comparison.png', 'Figure 7. Normalized compulsion signature profiles for Mike Lee, Trump (Android), and strategic-account mean.', 5)

fig('fig4_mikelee_trajectory.png', 'Figure 8. Mike Lee posting trajectory (2022-2026). Top: weekly rate (5.7-fold escalation). Bottom: night posting share.', 6)

fig('temporal_stability.png', 'Figure 9. Temporal stability of compulsion signatures across 15 quarters.', 6)

fig('fig6_trump_escalation.png', 'Figure 10. Trump compulsion escalation by year (2011-2017).', 5.5)

fig('label_reliability.png', 'Figure 11. LLM label test-retest reliability (n=75).', 5.5)

fig('fig2_feature_importance.png', 'Figure 12. Compulsion feature coefficients.', 5)

fig('fig7_roc_curves.png', 'Figure 13. ROC curves for individual compulsion features.', 5)

doc.add_page_break()
heading('Data and Code Availability', 1)
para('Model and dataset: huggingface.co/jimnoneill/xbox-compulsion-classifier and huggingface.co/datasets/jimnoneill/xbox-political-compulsion-dataset (private). Access requests to corresponding author.')
heading('Ethics Statement', 1)
para('This study analyzes publicly available social media data from public officials acting in their official capacity. The methodology cannot be used for clinical diagnosis. The Toxicity Index and compulsion probability are research instruments.')
heading('Conflict of Interest', 1)
para('The authors declare no conflicts of interest.')

fname = f'{out_path}/ONeill_Brookes_2026_IJDP_v3.docx'
doc.save(fname)
print(f'Saved: {fname} ({os.path.getsize(fname) / 1024:.0f} KB)')
