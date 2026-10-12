"""Search OpenAlex for relevant highly-cited papers."""
import requests, json

base = "https://api.openalex.org/works"

queries = [
    "social media addiction politicians temporal",
    "digital phenotyping social media compulsive behavior",
    "twitter toxicity political classification NLP",
    "Hawkes process social media temporal dynamics",
    "political polarization social media computational analysis",
    "behavioral addiction social media longitudinal study",
    "social media mental health digital phenotyping",
    "congressional twitter analysis political communication",
]

all_papers = []
seen_ids = set()

for q in queries:
    params = {
        "search": q,
        "filter": "from_publication_date:2022-01-01,is_oa:true,cited_by_count:>15",
        "sort": "cited_by_count:desc",
        "per_page": 10,
    }
    try:
        resp = requests.get(base, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for work in data.get("results", []):
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

                all_papers.append({
                    "title": work.get("title", ""),
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

print(f"Found {len(all_papers)} relevant open-access papers (2022+, >15 citations)\n")
for i, p in enumerate(all_papers[:25]):
    cites = p["citations"]
    auth = p["authors"][:55]
    yr = p["year"]
    title = p["title"][:110] if p["title"] else "No title"
    jrnl = p["journal"][:40] if p["journal"] else ""
    doi = p["doi"] or ""
    print(f"{i+1:2d}. [{cites:>4d} cites] {auth} ({yr})")
    print(f"    {title}")
    print(f"    {jrnl}  {doi}")
    print()

# Save
with open("/home/joneill/x_box/pipeline/output/relevant_papers.json", "w") as f:
    json.dump(all_papers[:30], f, indent=2)
print("Saved top 30 to output/relevant_papers.json")
