"""Generate corrected summary v3 for Joanna Brookes.
- Page 1-2: What changed, written for a non-scientist author, with each metric defined
- Page 3: Findings with full party counts, per-metric comparisons, Lee conjecture, notable individuals
"""
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
style.paragraph_format.space_after = Pt(4)
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

def metric_block(name, what_it_is, how_it_works, why_it_matters):
    p = doc.add_paragraph()
    r = p.add_run(name)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.bold = True
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(2)

    for label, text in [('What it is: ', what_it_is), ('How it works: ', how_it_works), ('Why it matters: ', why_it_matters)]:
        p2 = doc.add_paragraph()
        r1 = p2.add_run(label)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11)
        r1.italic = True
        r2 = p2.add_run(text)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(11)
        p2.paragraph_format.line_spacing = 1.4
        p2.paragraph_format.space_after = Pt(1)
        p2.paragraph_format.left_indent = Cm(0.5)
    doc.add_paragraph()  # spacer

# ============================================================
# TITLE
# ============================================================
para('Executive Summary: X-Box Compulsive Social Media Analysis Pipeline', bold=True)
para('Prepared for Joanna Brookes', italic=True)
para('Jamey O\u2019Neill, April 2026', italic=True)
doc.add_paragraph()

# ============================================================
# PAGES 1-2: What We Changed
# ============================================================
heading('What We Changed From the Original Prototype', 2)

para('Your prototype worked well as an exploration tool. You and your student built a custom ChatGPT that a user could feed a spreadsheet of tweets, and it would compute behavioral metrics and produce a report, all within a single chat session using Code Interpreter. It found real patterns in the Mike Lee archive: the extreme posting frequency, the 40% night share, the 99.7 compulsion score. Those findings held up when we rebuilt the analysis with more rigorous methods.')

para('The changes we made fall into three categories: replacing shallow text analysis with proper classification models, replacing the ad hoc scoring formulas with a statistically validated model, and making the whole thing reproducible. Below is an explanation of each new metric, what it measures, and why it gives us something the prototype could not.')

heading('The Problem With Lexicon Sentiment', 3)
para('The original prototype counted positive and negative words in each tweet and produced a polarity score from -1 to +1. It then used rule-based pattern matching to flag sarcasm (looking for quotation marks or phrases like "yeah right"), hostility (negative polarity combined with emphatic punctuation), and ridicule (laughter tokens in negative context). These heuristics work for obvious cases but miss the way political speech actually operates. A tweet like "Why aren\u2019t leftists celebrating the first person of color to serve as FBI director?" contains no negative words and no sarcasm markers, but it is clearly provocative. The prototype scored it neutral. Our replacement classifiers, trained on 124 million tweets, correctly identify it as ironic and tribal.')

heading('New Temporal Metrics', 3)
para('The prototype computed posting frequency, gaps between posts, and night share. We kept those ideas but formalized them using measures from the academic literature that have known statistical properties and can be compared across studies.')

metric_block(
    'Burstiness (B)',
    'A single number between -1 and +1 that captures whether someone posts in rapid-fire bursts with long gaps between them, or at a steady regular pace.',
    'It compares the spread (standard deviation) of the gaps between posts to their average. If the spread is much larger than the average, posts are bursty. If the spread is small, posts are regular. The formula is B = (spread minus average) divided by (spread plus average).',
    'A B near -1 means someone (or their scheduling tool) is posting at clockwork intervals. A B near 0 means random timing, like someone checking in whenever they happen to think of it. A B above +0.3 means posting in intense bursts followed by silence, a pattern the complex-systems literature associates with compulsive behavior. Mike Lee scores +0.755. The average senator scores +0.17.'
)

metric_block(
    'Time-of-Day Entropy',
    'A measure of how spread out across the day someone\u2019s posting is, expressed in bits.',
    'It takes the 24 hours of the day, counts what fraction of total tweets landed in each hour, and computes the Shannon entropy of that distribution. If every hour has the same share, entropy is at its maximum of 4.58 bits. If all tweets land in a single hour, entropy is zero.',
    'A staffer scheduling posts during business hours will produce low entropy (around 3.0 to 3.3 bits). Someone who posts at all hours, including 3am and 5am, produces high entropy (above 4.0). This turned out to be the single strongest predictor of compulsive use in our model, with a coefficient of +1.258. It is stronger than night posting alone because it captures the whole-day pattern rather than just one time window.'
)

metric_block(
    'Hawkes Self-Excitation (n*)',
    'A measure of whether each post triggers additional posts, creating a self-reinforcing cascade.',
    'For each tweet, we count how many follow-up tweets appear within the next five minutes. The average across all tweets gives an estimate of the branching ratio n*. If n* is below 0.3, most posts are independent of each other. If n* approaches 1.0, the account is at the tipping point of a self-sustaining posting spiral.',
    'This metric captures the feedback loop that characterizes compulsive engagement: post, get a response, respond to the response, get another response, keep going. Strategic accounts with scheduled posts show low self-excitation because each post is planned independently. Mike Lee\u2019s n* of 0.984 means he is essentially always in a posting cascade.'
)

metric_block(
    'Night Posting Intensity',
    'The fraction of all tweets posted between midnight and 6am UTC.',
    'A simple count: tweets with timestamps in the 00:00 to 05:59 window divided by total tweets. If posting were uniform across all hours, the expected share would be 25%.',
    'Posting at 3am when you should be sleeping is one of the clearest behavioral signals of compulsive use, directly analogous to sleep disruption in other behavioral addictions. Staff-managed accounts average about 3.7% night posting. The compulsive accounts in our ground truth cohort average 22.9%. Mike Lee is at 40.2%.'
)

metric_block(
    'Weekend Posting Ratio',
    'The per-day posting rate on weekends divided by the per-day rate on weekdays.',
    'Separately compute tweets per Saturday/Sunday and tweets per Monday through Friday, then take the ratio. A ratio of 1.0 means identical weekend and weekday posting. Below 0.5 means much less posting on weekends.',
    'Staff-managed accounts typically drop off on weekends because comms teams work business hours. Someone who posts compulsively does not stop on Saturday. This was the weakest of our five metrics (coefficient +0.204), because some strategic accounts also post on weekends for campaign reasons.'
)

heading('New Text Classification Metrics', 3)
para('In addition to the temporal metrics above, we built six new text classifiers to detect dimensions of political speech that the prototype\u2019s word-counting approach missed entirely.')

metric_block(
    'Ragebait',
    'Whether a tweet is designed to provoke outrage or anger in the reader.',
    'A machine learning classifier trained on 4,121 tweets labeled by a large language model, then distilled into a lightweight model that runs in milliseconds. Each tweet receives a binary score (ragebait or not).',
    'Ragebait is the currency of compulsive political posting. It drives engagement, which drives algorithmic amplification, which rewards the poster with attention, completing the compulsive loop. 47.8% of Mike Lee\u2019s tweets were classified as ragebait.'
)

metric_block(
    'Tribal Signaling',
    'Whether a tweet uses us-versus-them framing, naming or dismissing an out-group.',
    'Same classifier architecture. Trained with examples from both parties: "leftists," "the radical right," "woke mob," "MAGA extremists" all score as tribal.',
    '57.7% of Mike Lee\u2019s tweets contained tribal signaling. Standard toxicity classifiers scored his account at only 19.2 out of 100 because tribal framing is not the same thing as hate speech. This is the biggest gap the prototype missed.'
)

metric_block(
    'Performative Outrage',
    'Whether the expressed outrage is theatrical (performed for audience effect) rather than a genuine response to a substantive harm.',
    'The classifier was trained with both-party examples and an explicit instruction to score Democrats and Republicans equally when they engage in the same behavior.',
    '52.3% of Lee\u2019s tweets were classified as performative outrage. This matters because performative outrage is not captured by sentiment analysis. It reads as "negative" in a sentiment model, but the distinction between genuine anger and political theater is exactly what the audience responds to.'
)

doc.add_page_break()

# ============================================================
# PAGE 3: Findings
# ============================================================
heading('Key Findings', 2)

para('Senator Mike Lee', bold=True)
para('Senator Lee\u2019s @BasedMikeLee account is the most extreme case in our dataset by a wide margin. Over 3.5 years he posted 49,415 tweets, averaging one every 38 minutes around the clock. Forty percent of his posts occurred between midnight and 6am. His Hawkes self-excitation ratio of 0.984 means that virtually every tweet triggers additional tweets within minutes, a self-sustaining cascade. His burstiness of +0.755 is more than four times the Senate average (+0.17), and his inter-tweet gap distribution follows a power law rather than a normal curve. His posting trajectory is escalating: weekly volume grew from 73 tweets per week in early 2022 to over 420 by late 2025, with sudden collapses followed by rebound surges that match the cusp catastrophe dynamics predicted by Recovery Viability Theory. Standard toxicity classifiers score his overall virulence at 19.2 out of 100, but our custom classifiers find 57.7% tribal signaling, 52.3% performative outrage, and 47.8% ragebait. He has found a register of provocation that existing content moderation tools cannot see. The overall picture is consistent with someone whose relationship to the platform has crossed from strategic communication into compulsive engagement, and whose output is becoming more voluminous and more provocative over time.')

para('Party Trends: Republicans (n=48) vs. Democrats (n=50)', bold=True)
para('The two parties do not differ significantly on any individual metric, but the distributions tell different stories. Republican senators average P(compulsive)=0.436 (\u00b10.288), Democrats average 0.398 (\u00b10.226). The difference is not statistically significant (t=+0.73, p=0.47). However, Republicans have more members at the extremes: 8 Republicans score above 0.80 compared to only 3 Democrats, and 11 Republicans score below 0.20 compared to 10 Democrats. Democrats cluster more heavily in the middle ranges, with 15 in the 0.40 to 0.60 band versus only 8 Republicans.')

para('On the individual temporal features, one near-significant difference emerged: Republicans show higher burstiness (mean B=+0.189 vs. +0.158, t=+1.79, p=0.077), suggesting more rapid-fire posting patterns on the Republican side. Democrats show slightly higher night posting intensity (10.9% vs. 9.1%, t=-1.58, p=0.12), though this falls short of significance. Time-of-day entropy (R: 3.74, D: 3.72), Hawkes self-excitation (R: 0.877, D: 0.878), and weekend ratio (R: 0.622, D: 0.606) are virtually identical across parties.')

para('The five highest-scoring Republicans are Rubio (0.968), Cornyn (0.947), Hagerty (0.921), Blackburn (0.884), and Cruz (0.869). This cluster tends toward senators who use Twitter as a primary channel for base mobilization, with confrontational personal brands that reward constant engagement. The five highest-scoring Democrats are Murphy (0.959), Blumenthal (0.874), Coons (0.839), Schumer (0.740), and Durbin (0.734). The Democratic compulsive cluster tends toward policy advocacy, real-time commentary, and leadership communication rather than the tribal provocation style more common on the Republican side. Murphy has publicly discussed his own heavy social media engagement, and his temporal signatures are consistent with personal management of a significant portion of his output.')

para('At the strategic end, the five lowest Republicans are Sullivan (0.081), McConnell (0.057), Risch (0.052), Hoeven (0.041), and Shelby (0.014). The five lowest Democrats are Feinstein (0.079), Tester (0.073), Sinema (0.071), Gillibrand (0.070), and Klobuchar (0.053). Both parties\u2019 strategic floors include older senators, low-profile members, and those with known staff-managed communications operations.')

para('Notable Individuals', bold=True)
para('Lisa Murkowski (R-AK, 0.836) is a notable outlier, scoring as highly compulsive despite her reputation as a moderate. Her temporal patterns suggest personal, always-on engagement that does not match the combative style of other high-scoring Republicans. Chuck Grassley (R-IA, 0.434) falls right at the ambiguous midpoint, consistent with reports that his daughter has confirmed he personally authors some of his notably idiosyncratic tweets ("u kno what") while staff handle others. In the House, preliminary analysis of 30 members found Rep. Joaquin Castro (D-TX, 0.843) and Rep. Thomas Massie (R-KY, 0.841) as the highest scorers. Massie\u2019s 24% night posting share is consistent with his reputation for personally managing his social media presence.')

para('Summary', bold=True)
para('The pipeline produces calibrated, reproducible compulsion estimates grounded in the digital phenotyping and behavioral addiction literature. It replaces the prototype\u2019s lexicon-based sentiment and heuristic tone detection with twelve trained classifiers, replaces the ad hoc scoring formula with a logistic regression validated against independent evidence (AUC=0.933, Spearman r=0.912), and runs as deterministic code rather than within a ChatGPT session. The full manuscript is formatted for the International Journal of Drug Policy. The model, data, and code are archived on HuggingFace under private repositories.')

fname = f'{out_path}/Summary_for_Brookes.docx'
doc.save(fname)
print(f'Saved: {fname} ({os.path.getsize(fname) / 1024:.0f} KB)')
