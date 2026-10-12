"""Generate corrected two-page summary for Joanna Brookes.
Fixed: accurately describes the original prototype based on actual project documents."""
import os, sys
try:
    from docx import Document
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-docx'], check=True)
    from docx import Document

from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

out_path = '/home/joneill/x_box/pipeline/manuscript'
doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.5

def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(0, 0, 0)

def para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = bold
    run.italic = italic
    p.paragraph_format.line_spacing = 1.5

# ============================================================
# TITLE
# ============================================================
para('Executive Summary: X-Box Compulsive Social Media Analysis Pipeline', bold=True)
para('Prepared for Joanna Brookes', italic=True)
para('Jamey O\u2019Neill, April 2026', italic=True)
doc.add_paragraph()

# ============================================================
# PAGE 1: What We Changed and Why
# ============================================================
heading('Methodological Approach: What We Changed and Why', 2)

para('The original prototype was a custom ChatGPT configured with a detailed system prompt that specified scoring formulas, safety language, and output formats. The workflow was straightforward: the student uploaded a CSV of exported tweets into a ChatGPT session, and the model used Code Interpreter to run Python and pandas directly in the conversation. It parsed timestamps, computed posting frequency metrics, ran lexicon-based sentiment analysis (counting positive and negative words to produce a polarity score), applied heuristic pattern-matching rules to flag sarcasm, hostility, and ridicule, and then combined everything into weighted composite scores on a 0-to-100 scale. The entire analysis happened within a single chat session, and the GPT produced the behavioral report as its output.')

para('This prototype was effective for exploration. It identified real patterns in the Mike Lee archive, including the extreme posting cadence (37.7 tweets per day), the 40% night posting share, and a compulsion-like pattern intensity of 99.7 out of 100. However, it has three structural limitations that prevent it from supporting the kind of claims a publication would need to make.')

para('First, the sentiment and tone detection are shallow. Counting positive and negative words does not capture the way political speech actually works. A tweet reading "Why aren\u2019t leftists celebrating the first person of color to serve as FBI director?" contains no negative words but is clearly sarcastic provocation. The lexicon approach scored it neutral. Rule-based heuristics for sarcasm (looking for quotation marks or phrases like "yeah right") and hostility (negative polarity combined with emphatic punctuation) catch some of these cases but miss the majority. Political provocation operates through tribal framing, rhetorical questions, and performative outrage, none of which map cleanly onto a positive/negative word list.')

para('Second, the scoring formulas are ad hoc. The prototype assigned 25% weight to activity intensity, 25% to burstiness, 15% to night activity, and so on. These weights were chosen by judgment, not estimated from data, and there is no way to assess whether a score of 63.6 on "viral rage" means anything relative to a population. There is no calibration, no confidence interval, and no external validation.')

para('Third, the entire pipeline runs inside a ChatGPT session, which means it cannot be reproduced exactly. Code Interpreter executes code on the fly with no persistent state between sessions. Running the same analysis tomorrow could produce different numerical results because the model\u2019s code generation is stochastic, even when the underlying analysis logic is deterministic.')

para('We rebuilt the pipeline from the ground up to address each of these problems.')

para('For sentiment and tone, we replaced the lexicon approach with twelve classification models running in parallel. Six are established community models from the CardiffNLP group, trained on 124 million tweets, covering sentiment, emotion, offensiveness, irony, hate speech, and toxicity. Six more we trained ourselves on dimensions the standard models are blind to: ragebait, tribal signaling, performative outrage, epistemic manipulation, engagement baiting, and agency language. Training labels were generated by a large language model applied to 4,121 tweets from 14 accounts across both parties, then distilled into lightweight classifiers that produce deterministic outputs. These classifiers run identically every time on the same input.')

para('For the behavioral scoring, we replaced the ad hoc weights with a logistic regression fitted to a ground truth cohort of 32 accounts, where labels were constructed from evidence that is completely independent of the temporal features: staff testimony, documented incidents, health and demographic factors, and knowledge of Senate communications structures. The resulting model achieves AUC=0.933, and its predicted probabilities correlate at Spearman r=0.912 with the independent evidence ratings. When it says a senator has a 70% probability of compulsive posting, that number is grounded in external validation, not a circular rediscovery of its own inputs.')

para('For reproducibility, the entire pipeline runs as deterministic Python code on a local machine with GPU acceleration. Every step produces identical output on every run. The code, trained models, and data are archived on HuggingFace under version-controlled private repositories.')

doc.add_page_break()

# ============================================================
# PAGE 2: Findings
# ============================================================
heading('Key Findings', 2)

para('Senator Mike Lee', bold=True)
para('Senator Lee\u2019s @BasedMikeLee account is the most extreme case in our dataset by a wide margin. Over 3.5 years he posted 49,415 tweets, averaging one every 38 minutes around the clock. Forty percent of his posts occurred between midnight and 6am. His Hawkes self-excitation ratio of 0.984 means that virtually every tweet he posts triggers additional tweets within minutes, a self-sustaining cascade that is the temporal fingerprint of a compulsive engagement loop. His inter-tweet gap distribution follows a power law rather than a normal or exponential distribution, a pattern associated in the complex-systems literature with bursty, addiction-like behavior. What makes Lee\u2019s case most revealing is the gap between what standard classifiers see and what our custom models detect. CardiffNLP toxicity models score his overall virulence at 19.2 out of 100. He rarely uses overtly hateful or abusive language. But 57.7% of his tweets employ tribal us-versus-them framing, 52.3% contain performative outrage, and 47.8% are structured as ragebait. He has found a register of provocation that existing content moderation tools do not recognize. His posting trajectory is escalating: weekly volume grew from 73 tweets per week in early 2022 to over 420 by late 2025, with sudden collapses followed by rebound surges that match the cusp catastrophe dynamics predicted by Recovery Viability Theory. The overall picture is consistent with someone whose relationship to the platform has crossed from strategic communication into compulsive engagement.')

para('Party Trends', bold=True)
para('Compulsion probabilities did not differ significantly by party. Republican senators averaged P=0.436, Democrats P=0.398 (\u03c7\u00b2=0.15, p>0.05). Both parties contain members at both extremes. Among Republicans, Rubio (0.968), Cornyn (0.947), Hagerty (0.921), Blackburn (0.884), and Cruz (0.869) scored highest. This cluster tends to include senators who use Twitter as a primary channel for base mobilization, with confrontational personal brands that reward constant engagement. Among Democrats, Murphy (0.959), Blumenthal (0.874), Coons (0.839), and Schumer (0.740) scored highest. Murphy has publicly discussed his own heavy social media use, and his temporal signatures are consistent with someone who personally manages a significant portion of his output. The Democratic high-compulsion cluster tends toward policy advocacy and real-time commentary rather than the tribal provocation style more common on the Republican side.')

para('Notable Individuals', bold=True)
para('At the strategic end, McConnell (0.057) and Klobuchar (0.053) represent prototypical staff-managed accounts: posting concentrated in business hours, near-zero night activity, and regular inter-post intervals consistent with scheduling tools. Feinstein (0.079), who was 88 to 90 during the study period with documented cognitive decline, functions as a control case confirming that our model correctly identifies obvious staff management. Shelby (0.014) is the single most strategic account, consistent with his age and his retirement in January 2023. Lisa Murkowski (0.836) is a notable outlier, scoring as highly compulsive despite her reputation as a moderate. Her temporal patterns suggest personal, always-on engagement that does not match the combative style of other high-scoring Republicans. In the House, preliminary analysis of 30 members found Rep. Joaquin Castro (D-TX, 0.843) and Rep. Thomas Massie (R-KY, 0.841) as the highest scorers. Massie\u2019s 24% night posting share is consistent with his reputation for personally managing his social media presence.')

para('Summary', bold=True)
para('The pipeline produces calibrated, reproducible compulsion estimates grounded in the digital phenotyping and behavioral addiction literature. It is validated against evidence that is independent of the features it uses, and it correctly identifies both extremes (clear compulsion and clear staff management) as well as the large ambiguous middle where most senators actually fall. The full manuscript is formatted for the International Journal of Drug Policy. The model, data, and code are archived on HuggingFace. We are reaching out to Wayne Kepner and the other co-authors of the Qualia and Recovery Viability Theory papers for feedback on the theoretical integration before final submission.')

fname = f'{out_path}/Summary_for_Brookes.docx'
doc.save(fname)
print(f'Saved: {fname} ({os.path.getsize(fname) / 1024:.0f} KB)')
