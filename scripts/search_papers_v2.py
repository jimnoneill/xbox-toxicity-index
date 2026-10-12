"""Search OpenAlex for narrowly relevant papers."""
import requests, json

base = "https://api.openalex.org/works"

# Much more specific queries
queries = [
    "politician twitter behavior compulsive posting",
    "congressional social media toxicity analysis",
    "political tweet classification hate speech incivility",
    "social media addiction behavioral temporal patterns",
    "digital phenotyping online behavior mental health prediction",
    "Hawkes process burstiness social media posting",
    "political polarization twitter NLP classification",
    "senator representative twitter engagement analysis",
    "social media compulsive use disorder criteria",
    "online behavior changepoint detection temporal",
    "tweet sentiment analysis political discourse toxicity",
    "social media addiction scale measurement",
]

all_papers = []
seen_ids = set()

for q in queries:
    # Lower citation threshold to catch niche papers
    params = {
        "search": q,
        "filter": "from_publication_date:2021-01-01,cited_by_count:>10",
        "sort": "relevance_score:desc",
        "per_page": 5,
    }
    try:
        resp = requests.get(base, params=params, timeout=15)
        if resp.status_code == 200:
            for work in resp.json().get("results", []):
                wid = work.get("id", "")
                if wid in seen_ids:
                    continue
                seen_ids.add(wid)

                auths = work.get("authorships", [])
                authors = ", ".join([a.get("author", {}).get("display_name", "")
                                     for a in auths[:3]])
                if len(auths) > 3:
                    authors += " et al."

                journal = ""
                pl = work.get("primary_location")
                if pl and pl.get("source"):
                    journal = pl["source"].get("display_name", "")

                title = work.get("title", "")
                # Filter: must be about social media + political or addiction
                title_lower = title.lower() if title else ""
                relevant = any(kw in title_lower for kw in [
                    "social media", "twitter", "tweet", "politi", "congress",
                    "addiction", "compuls", "toxic", "polariz", "incivil",
                    "hawkes", "phenotyp", "digital", "online behav",
                    "sentiment", "hate speech", "misinform",
                ])
                if not relevant:
                    continue

                all_papers.append({
                    "title": title,
                    "year": work.get("publication_year", ""),
                    "journal": journal,
                    "citations": work.get("cited_by_count", 0),
                    "authors": authors,
                    "doi": work.get("doi", ""),
                    "query": q[:40],
                })
    except Exception as e:
        print(f"Query failed: {q[:40]}... {e}")

all_papers.sort(key=lambda x: x["citations"], reverse=True)

print(f"Found {len(all_papers)} relevant papers\n")
for i, p in enumerate(all_papers[:25]):
    cites = p["citations"]
    auth = p["authors"][:55]
    yr = p["year"]
    title = p["title"][:110] if p["title"] else ""
    jrnl = p["journal"][:45] if p["journal"] else ""
    doi = p["doi"] or ""
    print(f"{i+1:2d}. [{cites:>4d} cites] {auth} ({yr})")
    print(f"    {title}")
    print(f"    {jrnl}  {doi}")
    print()

with open("/home/joneill/x_box/pipeline/output/relevant_papers_v2.json", "w") as f:
    json.dump(all_papers, f, indent=2)
print(f"Saved {len(all_papers)} papers to output/relevant_papers_v2.json")
