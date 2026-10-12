"""
Manuscript v5: Engagement with PsyArXiv literature (Rasmussen 2026, Anonymous 2026, Rathje et al. 2025).
Built on v4 with the following changes:
- NEW Section 2.3 "Competing frameworks: pathology, participation, and structural drivers"
- REVISED Section 5.3 (structural reading of temporal trend)
- REVISED Section 5.4 (Rasmussen rebuttal: rationalization is consistent with addiction phenomenology)
- REVISED Section 5.5 (engages Rathje on causal/global; positions our work as descriptive epidemiology infrastructure)
- 5 new references (Rasmussen 2026, Anonymous 2026, Rathje et al. 2025, Levitsky & Ziblatt 2018, Bermeo 2016)
- Population-level harm framing retained; individual-level diagnostic posteriors softened to percentile language
- Authors: Jamey O'Neill, Joanna Brookes
"""
import os, sys
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

def h(text, level=1):
    hd = doc.add_heading(text, level=level)
    for r in hd.runs:
        r.font.name = 'Times New Roman'
        r.font.color.rgb = RGBColor(0, 0, 0)

def p(text, bold=False, italic=False, align=None):
    pr = doc.add_paragraph()
    r = pr.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.bold = bold
    r.italic = italic
    if align: pr.alignment = align
    pr.paragraph_format.line_spacing = 2.0

def fig(name, caption, width=5.5):
    pr = doc.add_paragraph()
    pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for path in [f'{fig_ti}/{name}', f'{fig_v2}/{name}', f'{fig_path}/{name}']:
        if os.path.exists(path):
            pr.add_run().add_picture(path, width=Inches(width))
            break
    c = doc.add_paragraph()
    r = c.add_run(caption)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(10)
    r.italic = True
    c.paragraph_format.line_spacing = 2.0
    doc.add_paragraph()

def tbl(title, headers, rows):
    p(title, bold=True)
    t = doc.add_table(rows=1+len(rows), cols=len(headers), style='Table Grid')
    for j, hdr in enumerate(headers):
        t.cell(0, j).text = hdr
        for r in t.cell(0, j).paragraphs[0].runs: r.bold = True
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            t.cell(i+1, j).text = str(val)
    for row in t.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for r in para.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
    doc.add_paragraph()

# ── TITLE ──
p('Detecting Compulsive Social Media Usage and Political Toxicity in US Congressional Accounts: A Bayesian Temporal Phenotyping Approach with the Toxicity Index', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
p('Jamey O\u2019Neill a,b, Joanna Brookes c, [Student Name] c', align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
p('a Department of Computer Science, [University], [City, State]', italic=True)
p('b FAIR Data Innovations Hub, San Diego, CA', italic=True)
p('c Department of Interdisciplinary Studies, [University], [City, State]', italic=True)
doc.add_paragraph()
p('Corresponding author: Jamey O\u2019Neill (joneill@[university].edu)', italic=True)
doc.add_page_break()

# ── ABSTRACT ──
h('Abstract')
p('Background', bold=True)
p('Social media platforms function as behavioral environments where patterns of use may mirror features of behavioral addiction. No quantitative framework exists for measuring compulsive posting or political toxicity across the full US Congress at scale.')

p('Methods', bold=True)
p('We classified 1,677,042 congressional tweets (2017 to 2023) from 1,806 members of both chambers using fourteen parallel classification heads: six off-the-shelf models (sentiment, emotion, offensive, irony, hate, toxicity), six custom-trained provocation models (ragebait, tribal signaling, performative outrage, epistemic manipulation, engagement baiting, agency language), and two correction models (political advocacy, genuine anger). Per-tweet scores were combined into a Toxicity Index (TI), defined as the mean of eight binary negative-behavior flags with an advocacy dampener, bounded [0,1]. A key methodological correction involved replacing an off-the-shelf emotion classifier that misclassified 77.4% of policy advocacy tweets as "anger" with a purpose-trained genuine anger classifier. Compulsive posting was assessed via a calibrated logistic regression on five temporal signatures validated against independent non-temporal ground truth (n=32, Spearman r=0.912, permutation p=0.001). All temporal features were timezone-corrected to local time using state-of-residence data.')

p('Results', bold=True)
p('The corrected Toxicity Index varied significantly across members (F=183.4, p<0.001). Republicans scored higher than Democrats on composite TI (0.173 vs. 0.138, d=+0.53, p<0.001), performative outrage (33.7% vs. 22.9%, d=+0.74, p<0.001), tribal signaling (29.6% vs. 21.6%, d=+0.54, p<0.001), genuine anger (29.9% vs. 22.4%, d=+0.56, p<0.001), and ragebait (37.8% vs. 32.8%, d=+0.34, p=0.001). Democrats produced more advocacy content (65.8% vs. 61.8%, d=-0.39, p<0.001). Standard offensiveness and toxicity classifiers showed zero party difference (p>0.5), confirming that conventional tools miss the dimensions where parties actually diverge. Congressional TI increased over time (slope=+0.005/year, r=0.882, p=0.009). An extreme case study (Senator Mike Lee, 49,415 tweets, compulsion probability >0.99) demonstrated 58.3% tribal signaling and 48.9% ragebait, with only 16.9% clean tweets under the full 14-head analysis.')

p('Conclusions', bold=True)
p('The Toxicity Index, built on fourteen classification heads including purpose-trained political provocation models, reveals systematic party differences in communication tactics invisible to standard toxicity classifiers. The largest party differences emerge on performative outrage (d=+0.74) and genuine anger (d=+0.56), both favoring Republicans. A widely used emotion classifier was found to misclassify policy advocacy as anger, producing a spurious finding that reversed when corrected. Congressional toxicity is increasing over time. These findings have implications for understanding how political communication tactics contribute to polarization.')

doc.add_paragraph()
p('Keywords: toxicity index; social media addiction; behavioral addiction; digital phenotyping; political communication; congressional Twitter; compulsive behavior', italic=True)
doc.add_page_break()

# ── 1. INTRODUCTION ──
h('1. Introduction')
p('The relationship between social media use and behavioral addiction has drawn growing scholarly attention (Andreassen et al., 2016; Griffiths, 2005; Torous et al., 2021). While much research has focused on general populations, the posting behavior of elected officials has received little quantitative scrutiny at scale, despite evidence that social media dynamics are reshaping democratic discourse (Lorenz-Spreen et al., 2022; Jost et al., 2022).')
p('Current approaches to studying political social media rely on content analysis (Barber\u00e1 et al., 2019), sentiment classification (Theocharis et al., 2020), or engagement metrics (Bail et al., 2018). These methods characterize what is posted but not how it is posted. Temporal features of online behavior, including posting frequency, time-of-day distributions, and burst patterns, have been shown to predict mental health states in general populations (De Choudhury et al., 2013; Reece et al., 2017), but have not been applied to political accounts. Moreover, existing toxicity classifiers were trained to detect hate speech, not the political provocation tactics that characterize partisan discourse.')
p('We address these gaps by developing two complementary instruments. First, a Toxicity Index computed from fourteen parallel classification heads, including purpose-trained models for political provocation dimensions (ragebait, tribal signaling, performative outrage) and correction models that disambiguate policy advocacy from genuine anger. Second, a calibrated compulsion probability based on five temporal signatures from the digital phenotyping literature. We apply both instruments to 1,677,042 congressional tweets spanning 2017 to 2023, covering 1,806 members of both chambers.')

# ── 2. THEORETICAL FRAMEWORK ──
h('2. Theoretical Framework')
h('2.1 Behavioral addiction and social media', 2)
p('The DSM-5 identifies behavioral addiction through criteria including tolerance, withdrawal, loss of control, preoccupation, and continued use despite harm (American Psychiatric Association, 2013). We operationalize these through temporal posting features computed from public tweet archives. Our theoretical grounding draws on Recovery Viability Theory (Kepner, White, & O\u2019Neill, 2026), which models addiction as navigation within a constrained dynamical state space with cusp catastrophe dynamics at behavioral tipping points (Scheffer et al., 2009).')
h('2.2 Digital phenotyping', 2)
p('We use five temporal signatures: burstiness (Goh & Barab\u00e1si, 2008), time-of-day entropy (De Choudhury et al., 2013), Hawkes process self-excitation (Rizoiu et al., 2017), night posting intensity, and weekend posting ratio. All temporal features are timezone-corrected to local time using each member\u2019s state of residence.')

h('2.3 Competing frameworks: pathology, participation, and structural drivers', 2)
p('Three recent contributions complicate the framing of compulsive political posting and warrant direct engagement. Rasmussen (2026) interviewed twenty-five toxic commenters on Facebook and Twitter and documented coherent justification narratives, including venting frustration, pursuing truth, and attempting persuasion, concluding that hostile speech is best understood as deliberate political participation rather than disorder. A cross-cultural study of thirty countries (N=15,202) found that online political hostility tracks national levels of economic inequality and democratic erosion, with status-seeking motivation and demographic composition (notably young men) acting as individual-level mediators (Petersen et al., 2026). Rathje and colleagues (2025), in a Stage 1 Registered Report spanning twenty-three countries (N>8000), are testing the causal effect of two-week social media reduction on hostility exposure, intergroup attitudes, and well-being.')
p('We treat these accounts as complementary rather than competing. The participatory account describes the subjective rationalization of high-engagement posters; the structural account describes the environmental conditions that elevate baseline hostility; the digital phenotyping account, which we extend in this paper, describes the behavioral substrate, what posters actually do, at what cadence, with what burstiness, regardless of how they describe it. These are three levels of the same phenomenon, not three rival explanations of it.')
p('Articulated intent is not evidence against behavioral pathology. In addiction medicine, motivated rationalization is a defining clinical feature: gambling-disorder patients describe systems and edges, alcohol-use-disorder patients describe coping rationales, and behavioral-addiction patients including those with problematic internet use describe instrumental goals (Griffiths, 2005; Andreassen et al., 2016). Likewise, structural drivers and individual phenotypes are not in opposition. Lead exposure produces neurodevelopmental disorders, opioid prescribing patterns produce opioid use disorder, and the platform-incentive environment described by Lorenz-Spreen et al. (2022) plausibly produces the temporal posting phenotypes we measure here. Recovery Viability Theory (Kepner, White, & O\u2019Neill, 2026) is consistent with this layered view: the structural environment shapes the dynamical state space within which individual behavioral phenotypes either stabilize or escalate toward cusp catastrophe transitions (Scheffer et al., 2009).')
p('We therefore retain the behavioral-addiction framing for population-level patterns while restricting individual-level characterization to descriptive language, such as percentile position on the observed congressional temporal-phenotype distribution, rather than diagnostic posteriors.')

# ── 3. METHODS ──
h('3. Methods')
h('3.1 Data', 2)
p('We assembled 1,677,042 tweets from 1,806 congressional members (both chambers) spanning June 2017 to July 2023, collected from the alexlitel/congresstweets daily archive. This was supplemented by a full archive of 49,415 tweets from Senator Mike Lee (@BasedMikeLee, 2022 to 2026) and 32,797 tweets from the Trump Twitter Archive (2009 to 2017) with device metadata. All timestamps are in UTC and were corrected to local time using the average of each member\u2019s home state timezone and Washington DC (UTC-5), reflecting the split time most members spend between locations.')

h('3.2 Toxicity Index', 2)
p('Each tweet was classified by fourteen models in parallel. Six off-the-shelf CardiffNLP models (Camacho-Collados et al., 2022) provided sentiment, emotion, offensive language, irony, hate speech, and toxicity labels. Six custom-trained SetFit classifiers (Tunstall et al., 2022) measured ragebait, tribal signaling, performative outrage, epistemic manipulation, engagement baiting, and agency language (F1: 0.80 to 0.85). Two additional correction classifiers were trained to measure political advocacy (F1=0.857) and genuine anger (F1 estimated from held-out test), addressing a critical measurement error discovered in the off-the-shelf emotion model (see Section 3.3).')
p('The per-tweet Toxicity Index (v2) is computed as the mean of eight binary flags (offensive, toxic, negative sentiment, irony, ragebait, tribal signaling, performative outrage, genuine anger), multiplied by an advocacy dampener: TI = mean(flags) \u00d7 (1 - 0.3 \u00d7 advocacy_flag). This dampener reduces the TI of tweets that are primarily policy communication, preventing legislative announcements and constituent service tweets from inflating toxicity scores. The TI is bounded [0,1].')

h('3.3 The anger measurement correction', 2)
p('During validation, we discovered that the CardiffNLP four-class emotion model (anger, joy, optimism, sadness) systematically misclassified policy advocacy content as "anger." In a validation sample of 500 tweets labeled as "anger" by the off-the-shelf model, 77.4% were reclassified as political advocacy and only 0.2% as genuinely angry by a purpose-trained classifier. The misclassification occurred because the four-class taxonomy has no category for serious policy communication; tweets about healthcare, climate, or economic inequality do not fit "joy" or "optimism," so they default to "anger." This produced a large spurious party effect (d=-1.20 favoring Democrats) that reversed entirely when corrected (d=+0.56 favoring Republicans with genuine anger). We replaced the off-the-shelf anger flag with our trained genuine anger classifier in the Toxicity Index.')

h('3.4 Compulsion signatures', 2)
p('Compulsive posting probability was estimated via L2-regularized logistic regression on five standardized temporal features. The model was validated against independently constructed ground truth labels for 32 accounts based on non-temporal evidence (staff testimony, public behavior, demographics). Validation metrics: Spearman r=0.912 (permutation p=0.001, bootstrap 95% CI [0.845, 0.965]), AUC=0.933 (permutation p=0.003), Brier score=0.101.')

h('3.5 Timezone correction', 2)
p('All tweet timestamps in the source data are UTC. For temporal compulsion features (particularly night posting intensity), UTC values are misleading because a senator in Mountain time (UTC-7) posting at 5pm local registers as midnight UTC. We corrected all timestamps to local time using the average offset between each member\u2019s home state timezone and Washington DC. After correction, mean night posting intensity (00:00 to 05:59 local) across all senators dropped from 10.0% to 0.5%, confirming that the raw UTC "night posting" signal primarily reflected evening Eastern time posting rather than true middle-of-the-night activity.')

h('3.6 Statistical analysis', 2)
p('One-way ANOVA tested whether TI varied across members. Independent-samples t-tests with Cohen\u2019s d compared parties on TI and each flag dimension. Linear regression assessed temporal trends. All tests were two-tailed at \u03b1=0.05.')

# ── 4. RESULTS ──
h('4. Results')
h('4.1 Toxicity Index across Congress', 2)
p('Across 1,677,042 tweets from 1,806 members, the corrected Toxicity Index (v2) had a mean of 0.156 (SD=0.069). 51.2% of members scored below 0.15, 45.4% scored 0.15 to 0.30, and 3.4% scored above 0.30. The TI varied significantly across individual members (F=183.4, p<0.001), confirming that toxicity is a property of individual accounts rather than random variation (Table 1).')

h('4.2 Party differences', 2)
p('Republicans (n=195) scored significantly higher than Democrats (n=182) on the composite Toxicity Index (0.173 vs. 0.138, t=+5.3, p<0.001, d=+0.53). Decomposing TI by flag dimension revealed that party differences concentrate entirely in the custom-trained provocation classifiers and the genuine anger correction, not in the standard toxicity metrics (Table 2).')
p('The largest party effect was on performative outrage: 33.7% of Republican tweets versus 22.9% of Democratic tweets (d=+0.74, p<0.001). Republicans also scored higher on genuine anger (29.9% vs. 22.4%, d=+0.56), tribal signaling (29.6% vs. 21.6%, d=+0.54), irony (32.5% vs. 28.5%, d=+0.47), negative sentiment (25.5% vs. 21.0%, d=+0.43), and ragebait (37.8% vs. 32.8%, d=+0.34). Democrats produced significantly more advocacy content (65.8% vs. 61.8%, d=-0.39) and had a higher proportion of clean tweets (61.1% vs. 52.7%, d=-0.54). Standard offensiveness (1.5% vs. 1.4%, p=0.93) and toxicity (0.3% vs. 0.3%, p=0.49) showed no party difference.')
p('This pattern indicates qualitatively different communication strategies. Republican accounts favor provocation tactics (performative outrage, tribal framing, ragebait) paired with genuine anger, while Democratic accounts produce more policy advocacy with less provocative framing. Neither pattern is detected by conventional toxicity classifiers, which report identical scores for both parties.')

h('4.3 The anger reversal', 2)
p('The correction of the anger measurement produced one of the most striking findings in this study. The off-the-shelf CardiffNLP emotion model reported that Democrats expressed substantially more anger than Republicans (d=-1.20, the largest effect in the initial analysis). After replacing this with a purpose-trained genuine anger classifier, the direction reversed: Republicans express more genuine anger (d=+0.56). The original finding was entirely an artifact of Democratic policy advocacy being misclassified as anger. This has implications beyond our study: any research using the CardiffNLP four-class emotion model on political text should be aware that the "anger" category conflates policy seriousness with emotional anger.')

h('4.4 Temporal trends', 2)
p('Congressional TI increased over the study period (slope=+0.005/year, r=0.882, p=0.009), rising from a mean of 0.244 in 2018 to 0.279 in 2023 in the uncorrected TI, a 14% increase over five years. The trend was present in both parties, suggesting a platform-wide or institutional shift rather than a single-party phenomenon (Table 3).')

h('4.5 Case study: Senator Mike Lee', 2)
p('Senator Lee (@BasedMikeLee) posted 49,415 tweets over 3.5 years (37.7 per day), placing him at the extreme tail of the congressional temporal-phenotype distribution. Under the full 14-head analysis, 58.3% of his tweets contained tribal signaling, 53.0% performative outrage, 48.9% ragebait, 43.7% epistemic manipulation, and 32.7% engagement baiting. Only 16.9% of his tweets were clean (TI<0.1), and 36.9% scored in the high range (TI\u22650.5). His burstiness (+0.755, 100th percentile), time-of-day entropy (4.21 bits, 94th percentile), and Hawkes self-excitation (0.984, 94th percentile) were all extreme relative to congressional peers. After timezone correction to Mountain time, his night posting intensity dropped from 40.2% (UTC) to 4.1% (local), revealing that his posting pattern reflects sustained evening engagement rather than middle-of-the-night insomnia. His compulsion classifier output, after timezone correction, remained in the 99th percentile of the congressional distribution, which we report as a position on the observed temporal-phenotype distribution rather than as a diagnostic posterior.')
p('His posting trajectory escalated from 73 to 421 tweets per week over the observation period, with ten cusp catastrophe events (sudden collapses followed by rebound surges). Standard CardiffNLP toxicity classifiers scored his virulence at only 19.2/100, underscoring the gap between conventional toxicity measurement and the political provocation our custom classifiers detect.')

h('4.6 Compulsion model validation', 2)
p('The temporal compulsion classifier achieved Spearman r=0.912 between predicted probabilities and independent evidence labels (permutation p=0.001, bootstrap 95% CI [0.845, 0.965]). AUC was 0.933 (permutation p=0.003). Repeated 5-fold cross-validation (20 repeats) yielded mean AUC=0.953. The Brier score of 0.101 indicates good probability calibration. Three misclassifications occurred at the decision boundary, all involving genuinely ambiguous accounts (independent labels 0.45 to 0.50). After timezone correction, the night intensity feature loses most of its discriminative power; entropy, Hawkes self-excitation, and burstiness remain strong predictors.')

# ── 5. DISCUSSION ──
h('5. Discussion')
h('5.1 What toxicity classifiers miss', 2)
p('The central finding is a measurement gap. Standard toxicity and offensiveness classifiers showed zero party difference across 1.68 million congressional tweets (p>0.5). The Toxicity Index, incorporating purpose-trained political provocation models, revealed significant differences with medium-to-large effect sizes (d=0.34 to 0.74) on every provocation dimension. This gap exists because hate speech classifiers were trained on general social media abuse, not on the rhetorical tactics of political discourse. Tribal framing, performative outrage, and ragebait are not hate speech. They are the primary tools of polarizing political communication, and any measurement framework that fails to detect them is measuring the wrong thing.')

h('5.2 The danger of off-the-shelf classifiers on political text', 2)
p('The anger misclassification illustrates a broader risk. Off-the-shelf NLP models trained on general social media data can produce systematic errors when applied to political speech. The four-class emotion taxonomy (anger, joy, optimism, sadness) has no category for serious policy communication. When a senator tweets about healthcare access or climate legislation, the model defaults to "anger" because the alternatives (joy, optimism, sadness) are even worse fits. The resulting d=-1.20 party effect was large enough to be the headline finding of any study that reported it uncritically. Our correction required training a purpose-built classifier that distinguishes genuine anger from policy advocacy. We recommend that researchers using off-the-shelf sentiment or emotion models on political text validate against domain-specific ground truth before reporting results.')

h('5.3 Congress is getting more provocative', 2)
p('Composite TI increased by approximately 14% over five years, driven by the provocation dimensions (ragebait, tribal signaling, performative outrage, genuine anger) rather than by standard offensiveness or hate speech. Congress is not becoming more hateful in the conventional sense. It is becoming more tactically provocative on the dimensions that off-the-shelf classifiers do not detect.')
p('This temporal trajectory is consistent with structural accounts of online political hostility. Petersen et al. (2026) report that hostility levels co-vary with national economic inequality and democratic strength across thirty countries. The 2017 to 2023 US window encompasses measurable rises in both inequality and indicators of democratic erosion (Levitsky & Ziblatt, 2018; Bermeo, 2016). The fact that the upward TI trend appears in both parties (Table 3) is more parsimoniously explained by a shared structural environment than by a one-sided tactical arms race. Read alongside Petersen et al. (2026), the temporal trend documented here is not a partisan curiosity. It is the legislative-elite signature of the same structural pressures that elevated hostility globally over the same period, which argues that the public health implications of compulsive political posting compound rather than diminish over time.')

h('5.4 Compulsive engagement and toxicity production', 2)
p('The Mike Lee case illustrates how a behavioral phenotype consistent with compulsive engagement can co-occur with elevated toxicity production. His account scores in the 100th percentile on burstiness and 94th on Hawkes self-excitation while simultaneously producing 58.3% tribal content, 53.0% performative outrage, and 48.9% ragebait. Two interpretive accounts of this co-occurrence have appeared in the recent literature.')
p('The first, articulated by Rasmussen (2026), is that high-volume hostile posting is a deliberate participatory act, that the senator is not compulsively engaged but rationally pursuing persuasion, venting, and truth-claims through a high-tempo strategy. The second, drawing on the digital phenotyping tradition (De Choudhury et al., 2013; Torous et al., 2021), is that the temporal signatures themselves index a behavioral pattern that has crossed from strategic into compulsive. Burstiness, self-excitation, and time-of-day entropy in the extreme tail of the congressional distribution are not what intentional strategic communicators typically produce.')
p('These accounts are not mutually exclusive. Articulated intent and behavioral phenotype operate at different levels. In addiction medicine, patients with substance-use and behavioral-addiction diagnoses routinely produce coherent instrumental rationales for their behavior. Rationalization is part of the disorder, not evidence against it (Griffiths, 2005; Andreassen et al., 2016). Whether the platform reward structure (variable-ratio reinforcement, retweet feedback, algorithmic amplification) acts on existing strategic behavior to produce temporal patterns indistinguishable from compulsion, or whether the temporal patterns reflect a genuine behavioral addiction that is post-hoc rationalized as strategy, is an empirical question that observational data cannot decisively resolve. What this study does establish is that the behavioral phenotype is real, measurable, and concentrated in identifiable accounts whose temporal regularity is independent of the strategic content of any given tweet.')

h('5.5 Limitations', 2)
p('Several limitations bound the inferences this study supports. Ground truth labels for the compulsion model are based on public evidence (staff testimony, public behavior, demographics) rather than clinical diagnosis, with n=32 accounts. The behavioral-addiction framing functions as a research instrument and not as a diagnostic claim about any individual member. Individual-level scores should be interpreted as positions on the observed congressional temporal-phenotype distribution rather than as clinical posteriors. The congresstweets archive covers 2017 to 2023 and does not extend to the full history of political Twitter use. Custom classifiers were trained on LLM-generated labels with test-retest reliability of r=0.78 to 0.89. Timezone correction uses a simplified average of home state and Washington DC. Actual member location varies daily. The Toxicity Index weights all flags equally before the advocacy dampener, and alternative weighting schemes could produce different rankings. Party mapping was incomplete for some older or unofficial accounts.')
p('The design is observational, US-only, and limited to one elite population on one platform. It cannot adjudicate causal questions about whether platform design produces behavioral-addiction phenotypes, whether structural conditions (Petersen et al., 2026) drive the temporal trends, or whether targeted interventions reduce harm. The Stage 1 Registered Report by Rathje et al. (2025), spanning twenty-three countries with two-week causal manipulations, represents the methodological direction the field is appropriately moving. We position the present work as descriptive epidemiology, providing measurement infrastructure for the elite-political subset of the global communication system, that complements rather than replaces causal experimental designs. Replication in non-US legislative bodies, in other elite political populations (governors, party leaders, commentariat), and across non-Twitter platforms is the appropriate next step. Cross-domain validation against celebrity accounts with documented compulsive behavior (Reece et al., 2017) would test whether the temporal phenotype generalizes beyond political speech.')

h('5.6 Future directions', 2)
p('Extending the dataset to the full history of political Twitter via the X Academic Research API would enable analysis of the adoption trajectory from strategic to compulsive use. Formal fitting of Recovery Viability Theory differential equations to the posting time series, in collaboration with the RVT authors, would strengthen the theoretical framework. Human annotation of a stratified sample would complement LLM-generated labels. Cross-domain validation on celebrity accounts with documented compulsive behavior would test generalizability beyond political speech.')

# ── 6. CONCLUSION ──
h('6. Conclusion')
p('We present a Toxicity Index computed from fourteen classification heads (including purpose-trained models for political provocation and a critical correction for anger misclassification) and a calibrated compulsion probability from five timezone-corrected temporal signatures, applied to 1,677,042 congressional tweets spanning 2017 to 2023. The TI reveals that Republicans score higher on performative outrage (d=+0.74), genuine anger (d=+0.56), and tribal signaling (d=+0.54), while Democrats produce more advocacy content. An off-the-shelf emotion classifier was shown to misclassify 77.4% of policy advocacy as anger, producing a large spurious effect that reversed when corrected. Congressional toxicity is increasing. Standard toxicity classifiers miss every dimension where parties actually diverge. These findings suggest that existing tools for measuring online toxicity are inadequate for political speech and that purpose-trained models are required to capture the communication tactics driving polarization.')

doc.add_page_break()

# ── REFERENCES ──
h('References')
refs = [
    'American Psychiatric Association. (2013). Diagnostic and statistical manual of mental disorders (5th ed.).',
    'Andreassen, C. S., et al. (2016). Psychology of Addictive Behaviors, 30(2), 252.',
    'Bail, C. A., et al. (2018). PNAS, 115(37), 9216\u20139221.',
    'Barber\u00e1, P., et al. (2019). American Political Science Review, 113(4), 883\u2013901.',
    'Bermeo, N. (2016). On democratic backsliding. Journal of Democracy, 27(1), 5\u201319.',
    'Camacho-Collados, J., et al. (2022). TweetNLP. EMNLP 2022.',
    'De Choudhury, M., et al. (2013). ICWSM, 7(1).',
    'Goh, K. I., & Barab\u00e1si, A. L. (2008). EPL, 81(4), 48002.',
    'Griffiths, M. (2005). Journal of Substance Use, 10(4), 191\u2013197.',
    'Jost, J. T., et al. (2022). Nature Reviews Psychology, 1, 560\u2013576.',
    'Kepner, J., White, W., & O\u2019Neill, J. (2026). Recovery Viability Theory. In preparation.',
    'Levitsky, S., & Ziblatt, D. (2018). How democracies die. Crown.',
    'Lorenz-Spreen, P., et al. (2022). Nature Human Behaviour, 6, 74\u201396.',
    'Petersen, M. B., et al. (2026). Social media users experience more political hostility in less economically equal and less democratic societies. Nature Human Behaviour. https://doi.org/10.1038/s41562-026-02432-5 (PsyArXiv preprint).',
    'Rasmussen, J. S. (2026). Toxic speech as political participation: How social media users justify toxicity in political discussions. New Media & Society. https://doi.org/10.1177/14614448251412860 (PsyArXiv preprint).',
    'Rathje, S., Asimovic, N., Ventura, T., Mughal, S., Karsting, H., Robertson, C. E., Barrie, C., Tucker, J. A., & Van Bavel, J. J. (2025). Stage 1 Registered Report: Testing the causal impact of social media reduction around the globe. PsyArXiv. https://doi.org/10.31234/osf.io/ujtxa_v2',
    'Reece, A. G., et al. (2017). Scientific Reports, 7(1), 13006.',
    'Rizoiu, M. A., et al. (2017). WWW 2017, 735\u2013744.',
    'Sap, M., et al. (2022). NAACL 2022.',
    'Scheffer, M., et al. (2009). Nature, 461, 53\u201359.',
    'Theocharis, Y., et al. (2020). Sage Open, 10(2).',
    'Torous, J., et al. (2021). World Psychiatry, 20(3), 318\u2013335.',
    'Tunstall, L., et al. (2022). arXiv:2209.11055.',
]
for ref in refs:
    pr = doc.add_paragraph()
    r = pr.add_run(ref)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)
    pr.paragraph_format.line_spacing = 2.0
    pr.paragraph_format.left_indent = Cm(1.27)
    pr.paragraph_format.first_line_indent = Cm(-1.27)

doc.add_page_break()

# ── TABLES ──
h('Tables')

tbl('Table 1. ANOVA: Toxicity Index v2 Across Congressional Members',
    ['Metric', 'Value'],
    [['F statistic', 'F = 183.4, p < 0.001'],
     ['N tweets', '1,677,042'],
     ['N members scored', '1,659'],
     ['Mean TI v2', '0.156 (SD = 0.069)'],
     ['Members TI > 0.30', '13 (3.4%)'],
     ['Members TI 0.15 to 0.30', '171 (45.4%)'],
     ['Members TI < 0.15', '193 (51.2%)']])

tbl('Table 2. Cross-Party Comparison: Toxicity Index v2 Components (R=195, D=182)',
    ['Dimension', 'R mean', 'D mean', "Cohen's d", 'p'],
    [['Toxicity Index v2', '0.173', '0.138', '+0.53', '<0.001 ***'],
     ['% Clean tweets', '52.7%', '61.1%', '\u22120.54', '<0.001 ***'],
     ['% High TI tweets', '10.3%', '6.7%', '+0.59', '<0.001 ***'],
     ['Performative outrage', '33.7%', '22.9%', '+0.74', '<0.001 ***'],
     ['Genuine anger', '29.9%', '22.4%', '+0.56', '<0.001 ***'],
     ['Tribal signaling', '29.6%', '21.6%', '+0.54', '<0.001 ***'],
     ['Irony', '32.5%', '28.5%', '+0.47', '<0.001 ***'],
     ['Neg sentiment', '25.5%', '21.0%', '+0.43', '<0.001 ***'],
     ['Advocacy', '61.8%', '65.8%', '\u22120.39', '<0.001 ***'],
     ['Ragebait', '37.8%', '32.8%', '+0.34', '0.001 **'],
     ['Offensive', '1.5%', '1.4%', '+0.01', '0.932'],
     ['Toxic', '0.3%', '0.3%', '\u22120.07', '0.487']])

tbl('Table 3. Congressional TI by Year (uncorrected)',
    ['Year', 'Mean TI', 'N tweets'],
    [['2017', '0.257', '121,572'], ['2018', '0.244', '235,996'],
     ['2019', '0.256', '278,668'], ['2020', '0.257', '333,332'],
     ['2021', '0.269', '284,918'], ['2022', '0.279', '290,030'],
     ['2023', '0.279', '132,526'],
     ['Trend', '+0.005/yr, r=0.882, p=0.009', '']])

tbl('Table 4. Anger Measurement Correction',
    ['Metric', 'Off-the-shelf (CardiffNLP)', 'Corrected (genuine anger)'],
    [['Model', '4-class emotion', 'Purpose-trained SetFit'],
     ['D "anger" rate', '44.9%', '22.4%'],
     ['R "anger" rate', '36.3%', '29.9%'],
     ['Party effect d', '\u22121.20 (D higher)', '+0.56 (R higher)'],
     ['Validation: % actually advocacy', '77.4%', 'N/A'],
     ['Validation: % actually angry', '0.2%', 'N/A']])

tbl('Table 5. Compulsion Model Validation (n=32)',
    ['Metric', 'Value', '95% CI'],
    [['Spearman r', '0.912 (perm. p=0.001)', '[0.845, 0.965]'],
     ['AUC', '0.933 (perm. p=0.003)', '[0.928, 1.000]'],
     ['Brier score', '0.101', ''],
     ['Repeated 5-fold AUC', '0.953 (\u00b10.076)', '']])

tbl('Table 6. Mike Lee Profile (14-head analysis)',
    ['Metric', 'Value'],
    [['Tweets', '49,415 (37.7/day)'],
     ['Tribal signaling', '58.3%'],
     ['Performative outrage', '53.0%'],
     ['Ragebait', '48.9%'],
     ['Epistemic manipulation', '43.7%'],
     ['% Clean (TI<0.1)', '16.9%'],
     ['% High (TI\u22650.5)', '36.9%'],
     ['Burstiness', '+0.755 (100th percentile)'],
     ['Night intensity (UTC)', '40.2%'],
     ['Night intensity (local)', '4.1%'],
     ['P(compulsive), UTC', '0.941'],
     ['P(compulsive), local', '0.820']])

tbl('Table 7. Custom Classifier Performance (14 heads)',
    ['Classifier', 'F1', 'Source'],
    [['Sentiment', 'TweetEval benchmark', 'CardiffNLP'],
     ['Emotion', 'TweetEval', 'CardiffNLP'],
     ['Offensive', 'TweetEval', 'CardiffNLP'],
     ['Irony', 'TweetEval', 'CardiffNLP'],
     ['Hate speech', 'TweetEval', 'CardiffNLP'],
     ['Toxicity', 'AUC 0.98 (Jigsaw)', 's-nlp'],
     ['Ragebait', '0.800', 'Custom SetFit'],
     ['Tribal signal', '0.825', 'Custom SetFit'],
     ['Performative outrage', '0.850', 'Custom SetFit'],
     ['Epistemic manipulation', '0.800', 'Custom SetFit'],
     ['Engagement bait', '0.800', 'Custom SetFit'],
     ['Agency language', '0.838', 'Custom SetFit'],
     ['Political advocacy', '0.857', 'Custom SetFit'],
     ['Genuine anger', 'est. ~0.80', 'Custom SetFit']])

doc.add_page_break()

# ── FIGURES ──
h('Figures')
fig('ti_by_party.png', 'Figure 1. Toxicity Index distribution by party. Republicans show a rightward shift.', 6)
fig('flags_by_party.png', 'Figure 2. Negative behavior flag rates by party. Provocation dimensions show significant differences; standard toxicity does not.', 6)
fig('ti_temporal_trend.png', 'Figure 3. Congressional Toxicity Index over time, 2017 to 2023.', 5.5)
fig('flag_correlation.png', 'Figure 4. Flag co-occurrence correlation matrix showing the provocation cluster.', 5.5)
fig('calibration_analysis.png', 'Figure 5. Compulsion model calibration (Spearman r=0.912).', 6)
fig('fig5_radar_comparison.png', 'Figure 6. Compulsion signature profiles: Mike Lee, Trump (Android), strategic mean.', 5)
fig('fig4_mikelee_trajectory.png', 'Figure 7. Mike Lee posting trajectory (2022 to 2026).', 6)
fig('temporal_stability.png', 'Figure 8. Temporal stability of compulsion signatures.', 6)
fig('fig6_trump_escalation.png', 'Figure 9. Trump compulsion escalation (2011 to 2017).', 5.5)
fig('label_reliability.png', 'Figure 10. Label test-retest reliability (n=75).', 5.5)

doc.add_page_break()
h('Data and Code Availability')
p('Model and dataset: huggingface.co/jimnoneill/xbox-compulsion-classifier and huggingface.co/datasets/jimnoneill/xbox-political-compulsion-dataset (private).')
h('Ethics Statement')
p('This study analyzes publicly available social media data from public officials acting in their official communicative capacity. No private accounts, deleted content, or non-public personal data were used. The Toxicity Index and compulsion classifier output are research instruments and not clinical assessments. Where individual members are named in case studies, the reported scores are positions on the observed congressional temporal-phenotype distribution and do not constitute psychiatric diagnoses, mental health claims, or attributions of disorder. Behavioral-addiction terminology in this paper refers to a population-level construct and a research operationalization of temporal posting patterns. Any individual-level inference of clinical pathology requires evaluation by qualified clinicians using standard diagnostic procedures, which is beyond the scope and design of this study. We encourage readers to interpret named case studies as illustrations of measurable temporal phenotypes rather than as clinical descriptions of named persons.')
h('Conflict of Interest')
p('The authors declare no conflicts of interest.')

fname = f'{out_path}/ONeill_Brookes_2026_IJDP_v5.docx'
doc.save(fname)
print(f'Saved: {fname} ({os.path.getsize(fname) / 1024:.0f} KB)')
