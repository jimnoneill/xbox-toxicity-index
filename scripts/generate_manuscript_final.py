"""
Generate the FINAL IJDP-formatted manuscript as .docx.
Changes from previous version:
- Authors: Jim O'Neill, Joanna Brooks, [Student Name]
- NO em dashes anywhere (restructure all sentences)
- NO references to labeling model (use "large language model" or "LLM")
- NO references to the preliminary 1.0 AUC model
- NO AI writing tells (no "importantly", "notably", "it is worth noting", etc.)
"""
import os, sys

try:
    from docx import Document
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-docx'], check=True)
    from docx import Document

from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

out_path = '/home/joneill/x_box/pipeline/manuscript'
fig_path = '/home/joneill/x_box/pipeline/figures'
fig_v2_path = '/home/joneill/x_box/pipeline/figures/v2'

doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
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
    for path in [f'{fig_path}/{name}', f'{fig_v2_path}/{name}']:
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

# ============================================================
# TITLE PAGE
# ============================================================
para('Detecting Compulsive Social Media Usage Patterns in US Congressional Accounts: A Bayesian Temporal Phenotyping Approach', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para('Jim O\u2019Neill a,b, Joanna Brooks c, [Student Name] c', align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para('a Department of Computer Science, [University], [City, State]', italic=True)
para('b FAIR Data Innovations Hub, San Diego, CA', italic=True)
para('c Department of Interdisciplinary Studies, [University], [City, State]', italic=True)
doc.add_paragraph()
para('Corresponding author: Jim O\u2019Neill (joneill@[university].edu)', italic=True)
doc.add_paragraph()
para('Word count: ~7,200')
para('Tables: 6')
para('Figures: 9')
doc.add_page_break()

# ============================================================
# ABSTRACT
# ============================================================
heading('Abstract', 1)

para('Background', bold=True)
para('Social media platforms increasingly function as behavioral environments where patterns of use may mirror features of behavioral addiction. While existing research has examined social media addiction in general populations, no quantitative framework exists for identifying compulsive posting patterns in public-facing political accounts, where such behavior may have outsized societal consequences through the amplification of divisive content.')

para('Methods', bold=True)
para('We developed a multi-modal classification pipeline combining (1) a calibrated logistic regression model using five compulsion signatures derived from digital phenotyping literature, including Goh-Barab\u00e1si burstiness, time-of-day entropy, Hawkes process self-excitation, night posting intensity, and weekend posting ratio; (2) twelve text classification heads measuring ragebait, tribal signaling, performative outrage, epistemic manipulation, engagement baiting, and agency language alongside standard sentiment and toxicity; and (3) a DSM-5-adjacent disorder criteria assessment adapted from Recovery Viability Theory. The model was validated against independently constructed ground truth labels based on non-temporal evidence (staff testimony, public statements, demographic factors) for 32 accounts using leave-one-out cross-validation, then applied to 98 US senators and 30 House members.')

para('Results', bold=True)
para('The calibrated classifier achieved AUC=0.933, F1=0.897, and accuracy=90.6% in leave-one-out cross-validation (n=32), with Spearman r=0.912 (p<0.0001) between predicted probabilities and independent evidence ratings. Time-of-day entropy (\u03b2=+1.258) and Hawkes self-excitation (\u03b2=+0.922) were the strongest predictors. Applied to the full Senate, the model produced calibrated probabilities (mean=0.417, SD=0.258) with 11 senators (11.2%) scoring above 0.80. Hold-out temporal validation on a 49,415-tweet archive demonstrated signature stability over 3.5 years (P(compulsive) drift of 1.7%). LLM-generated classification labels showed good test-retest reliability (Pearson r=0.78 to 0.89). Custom classifiers revealed that standard toxicity models missed 57.7% tribal signaling and 52.3% performative outrage in the highest-compulsion case. Preliminary House of Representatives analysis (n=30 members) revealed similar patterns, with compulsion probabilities ranging from 0.55 to 0.84.')

para('Conclusions', bold=True)
para('Temporal posting signatures provide a reliable, non-invasive method for estimating compulsive social media usage probability in political accounts. When validated against independent evidence, the approach yields calibrated probabilities with meaningful uncertainty, revealing that most accounts fall on a continuum rather than in discrete categories. The finding that standard toxicity classifiers miss the majority of political provocation content has implications for platform governance and democratic discourse.')

doc.add_paragraph()
para('Keywords: social media addiction; behavioral addiction; digital phenotyping; Bayesian classification; political communication; compulsive behavior; temporal analysis; Twitter/X', italic=True)
doc.add_page_break()

# ============================================================
# 1. INTRODUCTION
# ============================================================
heading('1. Introduction', 1)

para('The relationship between social media use and behavioral addiction has been the subject of growing scholarly attention (Andreassen et al., 2016; Griffiths, 2005). While much research has focused on general population usage patterns, the posting behavior of public figures, particularly elected officials, has received comparatively little quantitative scrutiny. This gap persists despite mounting evidence that some political accounts exhibit patterns consistent with compulsive use (Wells et al., 2020).')

para('Current approaches to studying political social media behavior typically rely on content analysis (Barber\u00e1 et al., 2019), sentiment classification (Theocharis et al., 2020), or engagement metrics (Bail et al., 2018). These methods characterize what is posted but not how it is posted. Specifically, the temporal dynamics of posting behavior that may reveal compulsive patterns have not been studied at scale. Research in digital phenotyping has demonstrated that temporal features of online behavior, including posting frequency, time-of-day distributions, and burst patterns, can predict mental health states including depression and anxiety (De Choudhury et al., 2013; Coppersmith et al., 2014; Reece et al., 2017). However, these methods have not been systematically applied to political accounts.')

para('Several conceptual gaps motivate this work. First, existing social media addiction scales (e.g., Bergen Social Media Addiction Scale; Andreassen et al., 2012) rely on self-report and cannot be applied to public accounts. Second, standard NLP toxicity classifiers were trained to detect offensive content, not the political provocation tactics (ragebait, tribal signaling, performative outrage) that compulsive political posters preferentially produce. Third, no existing framework connects temporal posting signatures to clinical addiction criteria in a formally validated manner.')

para('We address these gaps by developing a calibrated temporal phenotyping model that: (1) identifies five compulsion signatures from the digital phenotyping literature; (2) validates against independently constructed ground truth labels based on non-temporal evidence; (3) provides calibrated posterior probabilities with uncertainty estimates rather than binary classifications; (4) augments temporal analysis with twelve text classification heads capturing political provocation dimensions invisible to standard toxicity classifiers; and (5) maps findings onto DSM-5-adjacent behavioral addiction criteria through a formal disorder assessment framework inspired by Recovery Viability Theory (Kepner et al., 2026).')

# ============================================================
# 2. THEORETICAL FRAMEWORK
# ============================================================
heading('2. Theoretical Framework', 1)

heading('2.1 Behavioral addiction criteria and social media', 2)
para('The DSM-5 identifies behavioral addiction through criteria including tolerance (escalating usage to achieve the same effect), withdrawal (distress when unable to engage), loss of control (failed reduction attempts), preoccupation (excessive time allocation), and continued use despite harm (American Psychiatric Association, 2013). We operationalize these criteria through temporal posting features that can be computed from public tweet archives without self-report.')

heading('2.2 Recovery Viability Theory', 2)
para('Our theoretical framework draws on Recovery Viability Theory (RVT; Kepner, White, & O\u2019Neill, 2026), which applies viability theory from constrained dynamical systems to model addiction as navigation within a multi-dimensional state space. RVT provides several constructs applicable to social media compulsion. First, a logit-bounded state space where behavioral variables are naturally bounded on [0,1] via the sigmoid-logit transform, providing natural resistance to extremes (Aubin, 2011). Second, cusp catastrophe dynamics producing bistable regimes where sudden behavioral transitions occur at bifurcation boundaries (Zeeman, 1977). Third, critical slowing down, where variance increases and autocorrelation approaches unity as a system approaches a behavioral tipping point, providing early warning signals (Scheffer et al., 2009).')

heading('2.3 Digital phenotyping of temporal behavior', 2)
para('We build on established work in digital phenotyping (Torous et al., 2016; Insel, 2017) using five specific temporal signatures: burstiness (Goh & Barab\u00e1si, 2008), quantified as B = (\u03c3 \u2212 \u03bc)/(\u03c3 + \u03bc) of inter-event time distributions; time-of-day entropy (De Choudhury et al., 2013), the Shannon entropy of the hourly posting distribution; Hawkes process self-excitation (Rizoiu et al., 2017), the branching ratio n* capturing post-triggered posting cascades; night posting intensity, the proportion of posts during 00:00 to 05:59 UTC; and weekend posting ratio, the weekend-to-weekday posting rate.')

# ============================================================
# 3. METHODS
# ============================================================
heading('3. Methods', 1)

heading('3.1 Data sources', 2)
para('We assembled data from multiple sources: (1) Senator tweets (n=79,754) from the m-newhauser/senator-tweets dataset containing tweets from 99 US senators during 2021 (Newhauser, 2021); (2) a full tweet archive for Senator Mike Lee (@BasedMikeLee; n=49,415) spanning July 2022 to February 2026; (3) the Trump Twitter Archive (n=32,797) from 2009 to 2017, including device metadata enabling separation of personal (Android) from staff (iPhone) tweets (Brown, 2021); (4) a sample of congressional tweets (n=51,926) from the alexlitel/congresstweets daily archive, covering 313 House members across 26 sampled dates in 2023 to 2024; and (5) Twitter/X handles for 91 senators and 419 House members from the unitedstates/congress-legislators repository.')

heading('3.2 Compulsion signature extraction', 2)
para('For each account, we computed five temporal signatures. Burstiness parameter (B) following Goh and Barab\u00e1si (2008): B = (\u03c3 \u2212 \u03bc)/(\u03c3 + \u03bc) of inter-tweet intervals, ranging from \u22121 (regular) through 0 (Poisson) to +1 (maximally bursty). Time-of-day entropy (H): Shannon entropy of the 24-hour posting distribution, H = \u2212\u03a3 p(h) log\u2082 p(h), with maximum log\u2082(24) \u2248 4.58 bits. Night posting intensity (N): proportion of tweets posted 00:00 to 05:59 UTC, with an expected value under uniform posting of 0.25. Hawkes self-excitation ratio (n*): estimated branching ratio representing follow-up posts triggered within 5 minutes. Weekend posting ratio (W): ratio of per-day weekend to weekday posting rate.')

heading('3.3 Independent ground truth construction', 2)
para('A central methodological concern is avoiding tautological validation, the circularity of labeling accounts as "compulsive" based on temporal features and then reporting that temporal features predict those labels. To address this, we constructed ground truth labels for 32 accounts based exclusively on non-temporal evidence, rated on a continuous 0 to 1 scale (0=clearly strategic, 0.5=genuinely ambiguous, 1=clearly compulsive). Evidence sources included: staff testimony (2 accounts; Wolff, 2018; Woodward, 2018), public behavior and documented incidents (8 accounts), health and demographic factors (5 accounts), role-based inference from known communications structures (12 accounts), self-statements about usage (1 account), and mixed or ambiguous evidence (4 accounts). The resulting distribution (mean=0.44, with 19 of 32 accounts labeled 0.3 to 0.7) provides substantial mass in the ambiguous range, avoiding forced binary judgments.')

heading('3.4 Calibrated classification model', 2)
para('We employed L2-regularized logistic regression (C=1.0) on standardized features, as logistic regression naturally produces calibrated probability estimates and avoids artificially extreme posteriors. Validation used leave-one-out cross-validation on all 32 accounts. Model quality was assessed across multiple axes: classification accuracy and F1 (against binarized labels at 0.5), AUC (discrimination), Brier score (calibration), and Spearman and Pearson correlation between predicted probabilities and continuous independent labels.')

heading('3.5 Text classification pipeline', 2)
para('We deployed twelve text classification heads. Six off-the-shelf CardiffNLP models provided sentiment, emotion, offensive language, irony, hate speech, and toxicity classification (all RoBERTa-base, approximately 125M parameters, pre-trained on 124M tweets; Camacho-Collados et al., 2022). Six additional custom-trained SetFit few-shot classifiers (Tunstall et al., 2022) measured ragebait, tribal signaling, performative outrage, epistemic manipulation, engagement baiting, and agency language.')

para('Custom classifier training labels were generated by a large language model (671B parameters) applied to 4,121 tweets from 14 accounts across both parties. Each dimension received its own dedicated prompt with concrete examples of high and low scores for both parties to prevent partisan labeling bias. Soft labels were binarized and used to train SetFit classifiers with sentence-transformers/all-mpnet-base-v2 as backbone, achieving F1 scores of 0.80 to 0.85 on held-out test sets. Test-retest reliability was assessed by re-labeling a stratified sample of 75 tweets.')

heading('3.6 Temporal stability assessment', 2)
para('To assess whether compulsion signatures are stable over time, we split the 49,415-tweet Mike Lee archive into a training period (2022 to 2024; n=23,654) and hold-out period (2025 to 2026; n=25,761), computing signatures independently for each period. Per-quarter signatures across all 15 quarters of the archive provided additional trajectory data.')

heading('3.7 Disorder baseline and changepoint detection', 2)
para('We mapped temporal features onto eight DSM-5-adjacent addiction criteria and applied Bayesian Online Changepoint Detection (Adams & MacKay, 2007) to identify regime changes and cusp catastrophe events in posting rate time series.')

# ============================================================
# 4. RESULTS
# ============================================================
heading('4. Results', 1)

heading('4.1 Calibrated model validation', 2)
para('The logistic regression classifier achieved AUC=0.933, F1=0.897, and accuracy=90.6% in leave-one-out cross-validation on the 32-account cohort with independently constructed labels (Table 1). The Brier score of 0.101 indicates good probability calibration. Predicted probabilities correlated strongly with the continuous independent evidence ratings (Spearman r=0.912, p<0.0001; Pearson r=0.883, p<0.0001), demonstrating that the temporal signatures capture genuine variation in compulsive posting behavior as assessed by external evidence (Figure 1).')

para('Feature coefficients (Table 2) show time-of-day entropy (\u03b2=+1.258) and Hawkes self-excitation (\u03b2=+0.922) as the strongest predictors, followed by burstiness (\u03b2=+0.837) and night intensity (\u03b2=+0.584). Weekend ratio contributed minimally (\u03b2=+0.204). Three misclassifications occurred at the decision boundary, all involving accounts labeled as ambiguous (independent labels 0.45 to 0.50): Schumer (labeled 0.45, predicted 0.867), Warren (labeled 0.50, predicted 0.293), and Tuberville (labeled 0.50, predicted 0.160). These boundary errors reflect genuine disagreement between external evidence and temporal behavior, and all three accounts have plausible explanations for the discrepancy.')

heading('4.2 Senate-wide analysis', 2)
para('Applied to all 98 senators with sufficient data, the calibrated model produced a continuous distribution of compulsion probabilities (Figure 2) with mean=0.417, SD=0.258, and range 0.014 to 0.968. Eleven senators (11.2%) scored above 0.80, 35 (35.7%) fell in the ambiguous range (0.40 to 0.80), and 52 (53.1%) scored below 0.40 (Table 3). The highest-probability senators were Rubio (0.968), Murphy (0.959), Cornyn (0.947), Hagerty (0.921), and Blackburn (0.884). The lowest were Shelby (0.014), Hoeven (0.041), Risch (0.052), Klobuchar (0.053), and McConnell (0.057). Compulsive classification did not differ significantly by party (\u03c7\u00b2=0.15, p>0.05).')

heading('4.3 Preliminary House of Representatives analysis', 2)
para('Analysis of 30 House members with sufficient tweet volume from the congresstweets sample (Table 4) revealed compulsion probabilities ranging from 0.552 (Rep. Lois Frankel, D-FL) to 0.843 (Rep. Joaquin Castro, D-TX). The mean probability (0.678) was higher than the Senate mean (0.417), though this likely reflects sampling bias toward high-volume tweeters in the daily archive rather than a genuine chamber difference. Other high-compulsion House members included Rep. Thomas Massie (R-KY; P=0.841, night intensity=24.0%) and Rep. Don Bacon (R-NE; P=0.806, night intensity=17.2%).')

heading('4.4 Case study: Senator Mike Lee', 2)
para('Senator Mike Lee (@BasedMikeLee) represents an extreme outlier on all compulsion dimensions (Table 5, Figure 3). His 49,415 tweets over 1,311 days yield a posting rate of 37.7 tweets per day, with burstiness B=+0.755 (100th percentile among senators), time-of-day entropy H=4.21 bits (94th percentile), night posting intensity N=40.2% (100th percentile, 1.6 times the uniform baseline and 10.9 times the strategic-cohort mean), Hawkes self-excitation n*=0.984 (94th percentile), median response latency of 1.3 minutes, and a power-law inter-tweet gap distribution.')

para('Trajectory analysis (Figure 4) revealed an escalating pattern: early weekly posting rate of 73.5 tweets per week rising to 421.3 (a 5.7-fold increase, p<0.001). Ten cusp catastrophe events were detected. The largest was a collapse from 285 to 24 tweets per day on November 20, 2024 (z=\u22127.7), followed by a rebound surge.')

para('Standard CardiffNLP toxicity classifiers scored Lee\u2019s overall virulence at only 19.2/100. Custom classifiers revealed that 57.7% of tweets contained tribal signaling, 52.3% performative outrage, 47.8% ragebait, 43.1% epistemic manipulation, and 32.5% engagement baiting (Table 6). Only 33.6% used agentic language, with the majority employing victimhood framing. The enhanced virulence score incorporating both classifier sets was 31.9/100.')

heading('4.5 Temporal stability', 2)
para('Hold-out temporal validation demonstrated that compulsion signatures remain stable over time (Figure 5). Across the training (2022 to 2024) and hold-out (2025 to 2026) periods, P(compulsive) drifted by only 1.7% (0.939 to 0.923). Five of six features showed less than 4% change. Only burstiness showed meaningful drift (0.752 to 0.571, a 24% decrease), suggesting posting became more regular while remaining equally intense. Per-quarter analysis across 15 quarters showed P(compulsive) consistently between 0.87 and 0.93.')

heading('4.6 Trump escalation trajectory', 2)
para('The Trump Twitter Archive provided a natural experiment in compulsion escalation. P(compulsive) rose from 0.588 (2011) to 0.882 (2015, the campaign year), with night posting intensity increasing from 0.8% to 36.0% (Figure 6). Device metadata confirmed that the Android-only (personal) subset scored P=0.882 with 39.7% night intensity and a weekend ratio of 1.033, while the iPhone (staff) subset showed lower night intensity (22.7%), consistent with prior analyses (Robinson, 2016).')

heading('4.7 Label reliability', 2)
para('Test-retest reliability of LLM-generated classification labels was assessed by re-labeling a stratified sample of 75 tweets (Figure 7). Continuous score reliability was good: Pearson r=0.889 for ragebait, r=0.862 for tribal signaling, and r=0.777 for performative outrage (mean absolute error: 0.10 to 0.12). Cohen\u2019s kappa on binarized labels ranged from 0.479 (ragebait, moderate agreement) to 0.730 (tribal signaling, substantial agreement). The higher reliability of continuous scores relative to binary cutoffs supports the use of soft labels in training.')

# ============================================================
# 5. DISCUSSION
# ============================================================
heading('5. Discussion', 1)

heading('5.1 Principal findings', 2)
para('This study demonstrates that temporal posting signatures provide a reliable, non-invasive method for estimating the probability of compulsive social media usage in political accounts. When validated against independent non-temporal evidence, the calibrated logistic regression achieves AUC=0.933 and correlates at r=0.912 with external evidence ratings. The feature importance hierarchy is clinically coherent: time-of-day entropy, which captures whether an account posts uniformly across all 24 hours, is the strongest predictor. Hawkes self-excitation and burstiness follow, reflecting the escalation cascade characteristic of compulsive engagement loops. Night posting intensity, while informative, is less dominant when features are properly weighted together, suggesting that the overall pattern of always-on engagement matters more than any single time-of-day indicator.')

heading('5.2 Calibration and honest uncertainty', 2)
para('A methodological contribution of this work is the emphasis on calibrated probabilities rather than binary classifications. The model produces probabilities reflecting genuine uncertainty: Schumer at 0.87, Grassley at 0.43, Warren at 0.29. These middle-range values, and the three misclassifications at the boundary, are features of an honest model rather than deficiencies. The Brier score of 0.101 quantifies calibration quality. When the model says P=0.7, approximately 70% of such accounts should indeed exhibit compulsive patterns by external evidence standards. This calibration is essential for any practical application, whether in academic research or platform governance.')

heading('5.3 The toxicity-provocation gap', 2)
para('A key finding is the divergence between standard toxicity classification and custom political provocation classifiers. The most compulsive account in our study scored only 19.2/100 on CardiffNLP virulence but 40.3/100 on custom dimensions. This gap reflects a fundamental limitation of existing NLP models: they were trained to detect hate speech and abuse on general Twitter, not the political provocation tactics that characterize compulsive political posting. Tribal framing, rhetorical questions designed to provoke, and performative outrage about partisan grievances are linguistically distinct from toxic content but may be equally harmful to democratic discourse.')

heading('5.4 Implications for political polarization', 2)
para('The correlation between compulsive posting patterns and divisive content production raises questions about the role of individual behavioral pathology in political polarization. If provocative political content is disproportionately generated by individuals exhibiting compulsive-use signatures, then interventions targeting platform-level incentives alone may be insufficient without also addressing individual behavioral dynamics.')

heading('5.5 Theoretical contributions', 2)
para('Our application of Recovery Viability Theory to social media behavior represents, to our knowledge, the first formal connection between viability-theoretic addiction models and digital phenotyping. The RVT framework provides: logit-bounded state spaces for natural score constraints, cusp catastrophe dynamics explaining sudden posting surges and collapses, and critical slowing down as principled early warning signals.')

heading('5.6 Limitations', 2)
para('Several limitations warrant discussion. First, ground truth labels are based on public evidence and expert judgment rather than clinical diagnosis. Second, UTC timestamps may not reflect local time zones. Third, the senator dataset covers only 2021; longitudinal multi-year data would strengthen trajectory analysis. Fourth, House analysis is preliminary (n=30 members from sampled dates). Fifth, custom classifiers were trained on LLM-generated labels, introducing labeling model biases, though test-retest reliability was acceptable (r=0.78 to 0.89). Sixth, celebrity cross-domain validation was limited by data availability. Seventh, the logistic regression assumes linear decision boundaries; nonlinear models might capture feature interactions but risk overfitting on n=32.')

heading('5.7 Future directions', 2)
para('Next steps include: expanding House analysis to all 419 members with systematic tweet collection; cross-domain validation on accounts with documented compulsive use or addiction histories; human annotation of LLM-generated labels for debasing; hold-out temporal validation across additional accounts; retraining classifiers with larger embedding backbones; and formal integration of the changepoint model with disorder assessment for time-resolved compulsion trajectory analysis.')

# ============================================================
# 6. CONCLUSION
# ============================================================
heading('6. Conclusion', 1)
para('We present a validated computational framework for detecting compulsive social media usage patterns in political accounts through temporal behavioral phenotyping. When validated against independently constructed ground truth, the framework achieves AUC=0.933 and Spearman r=0.912 with external evidence ratings. The calibrated model reveals that most senators fall on a continuum of compulsive behavior (mean P=0.417, SD=0.258) rather than in discrete categories, with probabilities ranging from 0.01 to 0.97. Preliminary House analysis extends these findings to the lower chamber. The discovery that standard toxicity classifiers miss the majority of political provocation content produced by high-compulsion accounts has implications for understanding how individual behavioral pathology may contribute to political polarization.')

doc.add_page_break()

# ============================================================
# REFERENCES
# ============================================================
heading('References', 1)

refs = [
    'Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. arXiv preprint arXiv:0710.3742.',
    'American Psychiatric Association. (2013). Diagnostic and statistical manual of mental disorders (5th ed.). American Psychiatric Publishing.',
    'Andreassen, C. S., Torsheim, T., Brunborg, G. S., & Pallesen, S. (2012). Development of a Facebook addiction scale. Psychological Reports, 110(2), 501\u2013517.',
    'Andreassen, C. S., et al. (2016). The relationship between addictive use of social media and video games and symptoms of psychiatric disorders. Psychology of Addictive Behaviors, 30(2), 252.',
    'Aubin, J. P. (2011). Viability theory: New directions. Springer.',
    'Bail, C. A., et al. (2018). Exposure to opposing views on social media can increase political polarization. PNAS, 115(37), 9216\u20139221.',
    'Barber\u00e1, P., et al. (2019). Who leads? Who follows? American Political Science Review, 113(4), 883\u2013901.',
    'Brown, B. (2021). The Trump Archive. https://www.thetrumparchive.com/',
    'Camacho-Collados, J., et al. (2022). TweetNLP: Cutting-edge NLP for social media. EMNLP 2022.',
    'Coppersmith, G., Dredze, M., & Harman, C. (2014). Quantifying mental health signals in Twitter. CLPsych Workshop.',
    'De Choudhury, M., et al. (2013). Predicting depression via social media. ICWSM, 7(1).',
    'Goh, K. I., & Barab\u00e1si, A. L. (2008). Burstiness and memory in complex systems. EPL, 81(4), 48002.',
    'Griffiths, M. (2005). A \u2018components\u2019 model of addiction within a biopsychosocial framework. Journal of Substance Use, 10(4), 191\u2013197.',
    'Insel, T. R. (2017). Digital phenotyping: Technology for a new science of behavior. JAMA, 318(13), 1215\u20131216.',
    'Kepner, J., White, W., & O\u2019Neill, J. (2026). Recovery Viability Theory: A recursive framework for addiction recovery. Manuscript in preparation.',
    'Lam, L. T. (2014). Internet gaming addiction, problematic use of the internet, and sleep problems. Current Psychiatry Reports, 16(4), 444.',
    'Newhauser, M. (2021). Senator tweets dataset. HuggingFace. https://huggingface.co/datasets/m-newhauser/senator-tweets',
    'Reece, A. G., et al. (2017). Forecasting the onset and course of mental illness with Twitter data. Scientific Reports, 7(1), 13006.',
    'Rizoiu, M. A., et al. (2017). Expecting to be HIP: Hawkes intensity processes for social media popularity. WWW 2017, 735\u2013744.',
    'Robinson, D. (2016). Text analysis of Trump\u2019s tweets confirms he writes only the (angrier) Android half. Variance Explained.',
    'Scheffer, M., et al. (2009). Early-warning signals for critical transitions. Nature, 461(7260), 53\u201359.',
    'Theocharis, Y., et al. (2020). The dynamics of political incivility on Twitter. Sage Open, 10(2).',
    'Torous, J., et al. (2016). New tools for new research in psychiatry. JMIR Mental Health, 3(2), e16.',
    'Tunstall, L., et al. (2022). Efficient few-shot learning without prompts. arXiv:2209.11055.',
    'Wells, C., et al. (2020). How Trump drove coverage to the nomination. Political Communication, 33(4), 669\u2013676.',
    'Wolff, M. (2018). Fire and fury: Inside the Trump White House. Henry Holt.',
    'Woodward, B. (2018). Fear: Trump in the White House. Simon & Schuster.',
    'Zeeman, E. C. (1977). Catastrophe theory. Scientific American, 234(4), 65\u201383.',
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

make_table('Table 1. Calibrated Model Validation (Leave-One-Out Cross-Validation, n=32)',
    ['Metric', 'Value'],
    [['Accuracy', '90.6% (29/32)'], ['F1 Score', '0.897'], ['AUC', '0.933'],
     ['Brier Score', '0.101'], ['Spearman r', '0.912 (p<0.0001)'],
     ['Pearson r', '0.883 (p<0.0001)'], ['Misclassifications', '3 (all at decision boundary)']])

make_table('Table 2. Logistic Regression Feature Coefficients',
    ['Feature', '\u03b2', 'Interpretation'],
    [['Time-of-day entropy', '+1.258', 'Posting across all 24 hours'],
     ['Hawkes n*', '+0.922', 'Self-triggering posting cascades'],
     ['Burstiness (B)', '+0.837', 'Bursty vs. regular intervals'],
     ['Night intensity', '+0.584', 'Posting 00:00 to 05:59 UTC'],
     ['Weekend ratio', '+0.204', 'Weekend vs. weekday rate'],
     ['Intercept', '+0.253', '']])

make_table('Table 3. Senate-Wide Compulsion Probability Distribution (n=98)',
    ['Category', 'P(compulsive)', 'Count (%)'],
    [['High compulsion', '>0.80', '11 (11.2%)'], ['Elevated', '0.60 to 0.80', '14 (14.3%)'],
     ['Ambiguous', '0.40 to 0.60', '21 (21.4%)'], ['Low', '0.20 to 0.40', '28 (28.6%)'],
     ['Strategic', '<0.20', '24 (24.5%)'], ['Mean (SD)', '', '0.417 (0.258)']])

make_table('Table 4. House of Representatives Preliminary Analysis (Top 10)',
    ['Member', 'P(comp)', 'Night %', 'Tweets'],
    [['JoaquinCastrotx (D-TX)', '0.843', '31.9%', '204'],
     ['RepThomasMassie (R-KY)', '0.841', '24.0%', '196'],
     ['RepDonBacon (R-NE)', '0.806', '17.2%', '308'],
     ['RepYoungKim (R-CA)', '0.739', '19.5%', '133'],
     ['RepChipRoy (R-TX)', '0.734', '13.8%', '189'],
     ['RepDwightEvans (D-PA)', '0.727', '12.8%', '257'],
     ['FrankPallone (D-NJ)', '0.722', '17.2%', '122'],
     ['Jim_Jordan (R-OH)', '0.718', '17.0%', '188'],
     ['RepJamesComer (R-KY)', '0.708', '10.4%', '134'],
     ['RepEspaillat (D-NY)', '0.703', '10.3%', '379']])

make_table('Table 5. Compulsion Profile: Senator Mike Lee (@BasedMikeLee)',
    ['Metric', 'Value', 'Percentile'],
    [['Total tweets', '49,415', ''], ['Date range', 'Jul 2022 to Feb 2026', ''],
     ['Tweets/day', '37.7', '>99th'], ['Burstiness B', '+0.755', '100th'],
     ['Time entropy', '4.21 bits', '94th'], ['Night intensity', '40.2%', '100th'],
     ['Hawkes n*', '0.984', '94th'], ['Weekend ratio', '0.838', '90th'],
     ['Median gap', '2.5 min', '>99th'], ['P(compulsive)', '~1.000', '>99th']])

make_table('Table 6. Custom Classifier Results: Mike Lee Virulence Dimensions',
    ['Dimension', 'F1', 'Rate (5K)', 'Description'],
    [['Tribal signaling', '0.825', '57.7%', 'Us-vs-them framing'],
     ['Performative outrage', '0.850', '52.3%', 'Theatrical outrage'],
     ['Ragebait', '0.800', '47.8%', 'Designed to provoke'],
     ['Epistemic manipulation', '0.800', '43.1%', 'Cherry-picking'],
     ['Agency language', '0.838', '33.6%', 'Active vs. victimhood'],
     ['Engagement baiting', '0.800', '32.5%', 'Polls, CTAs'],
     ['CardiffNLP virulence', '', '19.2/100', 'Standard toxicity'],
     ['Enhanced virulence', '', '31.9/100', 'Combined score']])

doc.add_page_break()

# ============================================================
# FIGURES
# ============================================================
heading('Figures', 1)

fig('calibration_analysis.png', 'Figure 1. Calibration analysis. Left: LOO-predicted P(compulsive) vs. independent evidence labels (Spearman r=0.912). Right: Predicted probabilities by evidence category.', 6)
fig('senator_calibrated_distribution.png', 'Figure 2. Distribution of calibrated compulsion posteriors across 98 US senators (2021). Mean=0.417.', 5.5)
fig('fig5_radar_comparison.png', 'Figure 3. Normalized compulsion signature profiles for Mike Lee, Trump (Android), and the strategic-account mean.', 5)
fig('fig4_mikelee_trajectory.png', 'Figure 4. Mike Lee posting trajectory, 2022 to 2026. Top: weekly rate (5.7-fold escalation). Bottom: night posting share.', 6)
fig('temporal_stability.png', 'Figure 5. Temporal stability of compulsion signatures across 15 quarters, with train/hold-out split.', 6)
fig('fig6_trump_escalation.png', 'Figure 6. Trump compulsion escalation by year (2011 to 2017). P(compulsive) rises from 0.588 to 0.882.', 5.5)
fig('label_reliability.png', 'Figure 7. LLM label test-retest reliability (n=75). Pearson r ranges from 0.777 to 0.889.', 6)
fig('fig2_feature_importance.png', 'Figure 8. Feature importance: logistic regression coefficients.', 5)
fig('fig7_roc_curves.png', 'Figure 9. ROC curves for individual features on the 32-account ground truth cohort.', 5)

doc.add_page_break()
heading('Data and Code Availability', 1)
para('The model and dataset are available as private repositories on HuggingFace (jimnoneill/xbox-compulsion-classifier and jimnoneill/xbox-political-compulsion-dataset). Access requests may be directed to the corresponding author.')
heading('Ethics Statement', 1)
para('This study analyzes publicly available social media data from public officials acting in their official capacity. No private communications were accessed. The methodology cannot and should not be used for clinical diagnosis. Ground truth labels reflect research judgments about posting behavior, not clinical assessments.')
heading('Conflict of Interest', 1)
para('The authors declare no conflicts of interest.')

fname = f'{out_path}/ONeill_Brooks_2026_IJDP_CompulsiveSocialMedia.docx'
doc.save(fname)
print(f'Saved: {fname}')
print(f'Size: {os.path.getsize(fname) / 1024:.0f} KB')
