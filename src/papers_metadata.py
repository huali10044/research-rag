"""
papers_metadata.py — Structured metadata for Hua Li's research papers.

Add your full paper list here. Each entry is used to:
  1. Build a rich text blob for embedding
  2. Populate metadata fields for filtered retrieval
  3. Power the citation/source display in query results

If you have PDFs, drop them in /data/papers/ and they'll be ingested automatically.
The metadata entries complement PDFs by providing clean abstracts and keywords
even when the full PDF text is noisy (scanned pages, headers/footers, etc.).

pdf_filename (optional): if this paper is ALSO present as a raw PDF in
PAPERS_DIR, set this to that PDF's exact filename. ingest.py uses it to
recognize both representations as the same work and assign them a shared
canonical work_id, instead of indexing them as two separate papers with
different titles (a filename-derived one for the PDF, a clean one here).
See KI-1 in eval/README.md for why this matters.
"""

PAPERS = [
    {
        "title": "Adaptive Interest Modeling Improves Content Services at the Network Edge",
        "authors": [
            "Hua Li", "Ralph Costantini", "David Anhalt", "Rafael Alonso",
            "Mark-Oliver Stehr", "Carolyn Talcot", "Minyoung Kim",
            "Timothy McCarthy", "Samuel Wood"
        ],
        "year": 2014,
        "venue": "IEEE Military Communications Conference (MILCOM 2014)",
        # This paper is also ingested from a raw PDF (see PAPERS_DIR). pdf_filename
        # lets ingest.py recognize both representations as the same work and merge
        # them under one canonical work_id — see KI-1 in eval/README.md.
        "pdf_filename": "2014-Adaptive Interest Modeling Improves Content Services at the Network Edge-06956895.pdf",
        "doi": "10.1109/MILCOM.2014.175",
        "url": "http://dx.doi.org/10.1109/MILCOM.2014.175",
        "abstract": (
            "This paper presents an adaptive interest modeling system that improves content "
            "services delivered at the network edge. The system dynamically models user interests "
            "to prioritize and personalize content delivery in constrained military network "
            "environments, reducing bandwidth consumption while improving relevance of delivered "
            "content. The approach integrates behavioral signals and contextual information to "
            "update user profiles in real time."
        ),
        "keywords": [
            "interest modeling", "user profiling", "content services", "network edge",
            "personalization", "military communications", "adaptive systems", "DARPA"
        ],
        "summary": (
            "MILCOM 2014 paper on adaptive user interest modeling for military network edge "
            "computing. Part of DARPA CBMEN program. Demonstrates real-time profile updates "
            "to drive content prioritization in bandwidth-constrained environments."
        ),
    },
    {
        "title": "Opaque Attribute Alignment",
        "authors": ["Jennifer Sleeman", "Rafael Alonso", "Hua Li", "Art Pope", "Antonio Badia"],
        "year": 2012,
        "venue": "IEEE 28th International Conference on Data Engineering Workshops (ICDEW 2012)",
        "pdf_filename": "2012-Opaque Attribute Alignment-06313650.pdf",
        "doi": "10.1109/ICDEW.2012.62",
        "url": "http://dx.doi.org/10.1109/ICDEW.2012.62",
        "abstract": (
            "Presents a method for aligning opaque attributes across heterogeneous data sources "
            "where attribute semantics are not explicitly defined. The approach uses statistical "
            "and structural techniques to discover correspondences between attributes that lack "
            "explicit schema documentation, enabling data integration across disparate sources."
        ),
        "keywords": [
            "schema matching", "attribute alignment", "data integration",
            "heterogeneous data", "data engineering", "ontology alignment"
        ],
        "summary": (
            "ICDEW 2012 paper on aligning attributes across data sources with unknown or opaque "
            "semantics. Addresses schema matching without explicit metadata. Relevant to "
            "data federation and knowledge graph construction."
        ),
    },
    {
        "title": "Spatial Event Prediction by Combining Value Function Approximation and Case-Based Reasoning",
        "authors": ["Hua Li", "Héctor Muñoz-Avila", "Diane Bramsen", "Chad Hogg", "Rafael Alonso"],
        "year": 2009,
        "venue": "8th International Conference on Case-Based Reasoning (ICCBR 2009)",
        "doi": "10.1007/978-3-642-02998-1_33",
        "url": "http://dx.doi.org/10.1007/978-3-642-02998-1_33",
        "abstract": (
            "Proposes a hybrid approach for spatial event prediction that combines reinforcement "
            "learning value function approximation with case-based reasoning. The system learns "
            "from historical spatial event patterns and retrieves analogous past cases to predict "
            "future events in geographic space. Evaluated on real-world event datasets."
        ),
        "keywords": [
            "spatial event prediction", "case-based reasoning", "value function approximation",
            "reinforcement learning", "hybrid AI", "geospatial analytics", "IARPA"
        ],
        "summary": (
            "ICCBR 2009 paper combining RL and CBR for geospatial event prediction. "
            "Part of IARPA program work. Demonstrates hybrid reasoning under uncertainty "
            "for real-world predictive analytics tasks."
        ),
    },
    {
        "title": "Profile-Based Security Against Malicious Mobile Agents",
        "authors": ["Hua Li", "Glena Greene", "Rafael Alonso"],
        "year": 2006,
        "venue": "2nd International Conference on Advanced Data Mining and Applications (ADMA 2006)",
        "doi": "10.1007/11811305_111",
        "url": "http://dx.doi.org/10.1007/11811305_111",
        "abstract": (
            "Presents a profile-based security mechanism to detect and defend against malicious "
            "mobile agents in distributed computing environments. User and agent behavior profiles "
            "are built dynamically and used to identify anomalous activity patterns indicative "
            "of malicious intent, enabling adaptive access control."
        ),
        "keywords": [
            "mobile agents", "security", "user profiling", "anomaly detection",
            "distributed systems", "behavioral modeling", "access control"
        ],
        "summary": (
            "ADMA 2006 paper on using behavioral profiles to detect malicious mobile agents. "
            "Early work on profiling and anomaly detection applied to distributed system security."
        ),
    },
    {
        "title": "An Adaptive Nearest Neighbor Search for a Parts Acquisition ePortal",
        "authors": ["Rafael Alonso", "Jeffrey A. Bloom", "Hua Li", "Chumki Basu"],
        "year": 2003,
        "venue": "9th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2003)",
        "pdf_filename": "2003-An adaptive nearest neighbor search for a parts acquisition ePortal-p693-alonso.pdf",
        "doi": "10.1145/956750.956842",
        "url": "http://doi.acm.org/10.1145/956750.956842",
        "abstract": (
            "Describes an adaptive nearest neighbor search algorithm tailored for a parts "
            "acquisition ePortal. The system learns user preferences over time and adjusts "
            "similarity metrics to improve search relevance for industrial parts queries. "
            "Presented at KDD, one of the premier venues for machine learning and data mining."
        ),
        "keywords": [
            "nearest neighbor search", "adaptive similarity", "recommendation",
            "e-commerce", "information retrieval", "user modeling", "KDD"
        ],
        "summary": (
            "KDD 2003 paper on adaptive k-NN for an e-commerce parts search portal. "
            "Early work on learned similarity metrics and personalized information retrieval. "
            "Precursor to modern embedding-based retrieval systems."
        ),
    },
    {
        "title": "Lessons from the Implementation of an Adaptive Parts Acquisition ePortal",
        "authors": ["Rafael Alonso", "Jeffrey A. Bloom", "Hua Li"],
        "year": 2003,
        "venue": "12th ACM International Conference on Information and Knowledge Management (CIKM 2003)",
        "pdf_filename": "2003-Lessons from the implementation of an adaptive parts acquisition ePortal-p169-alonso.pdf",
        "doi": "10.1145/956863.956896",
        "url": "http://doi.acm.org/10.1145/956863.956896",
        "abstract": (
            "Reports practical lessons learned from deploying an adaptive parts acquisition "
            "portal at scale. Covers challenges in real-world deployment of adaptive retrieval "
            "systems, including cold-start problems, feedback collection, and system performance "
            "under production workloads."
        ),
        "keywords": [
            "adaptive systems", "information retrieval", "e-commerce", "deployment lessons",
            "production systems", "cold start", "user feedback", "CIKM"
        ],
        "summary": (
            "CIKM 2003 companion paper covering production deployment lessons for the adaptive "
            "ePortal. Valuable for practitioners deploying recommendation and retrieval systems."
        ),
    },
    {
        "title": "Model-Guided Information Discovery for Intelligence Analysis",
        "authors": ["Rafael Alonso", "Hua Li"],
        "year": 2005,
        "venue": "14th ACM International Conference on Information and Knowledge Management (CIKM 2005)",
        "pdf_filename": "2005-Model-guided information discovery for intelligence analysis-p269-alonso.pdf",
        "doi": "10.1145/1099554.1099621",
        "url": "http://doi.acm.org/10.1145/1099554.1099621",
        "abstract": (
            "Presents a model-guided approach to information discovery that supports intelligence "
            "analysts in navigating large document collections. The system uses structured "
            "domain models to guide search and surfacing of relevant information, reducing "
            "cognitive load for analysts working with heterogeneous data sources."
        ),
        "keywords": [
            "information discovery", "intelligence analysis", "knowledge models",
            "document retrieval", "CIKM", "analyst support", "structured search"
        ],
        "summary": (
            "CIKM 2005 paper on model-driven information retrieval for intelligence analysts. "
            "Early work on structured knowledge-guided search — conceptually related to "
            "modern knowledge-graph-augmented RAG architectures."
        ),
    },
    {
        "title": "Managing Analysis Context",
        "authors": ["Hua Li", "Rafael Alonso"],
        "year": 2012,
        "venue": "12th International Workshop on Web Information and Data Management (WIDM 2012)",
        "pdf_filename": "2012-Managing analysis context-p33-li.pdf",
        "doi": "10.1145/2389936.2389945",
        "url": "http://doi.acm.org/10.1145/2389936.2389945",
        "abstract": (
            "Addresses the problem of maintaining and leveraging analysis context across "
            "interactive analytical sessions. The system tracks the history and focus of "
            "an analyst's investigation and uses this context to improve the relevance of "
            "retrieved information and suggested next steps in exploratory analysis workflows."
        ),
        "keywords": [
            "analysis context", "context management", "information retrieval",
            "exploratory analysis", "session modeling", "knowledge management", "WIDM"
        ],
        "summary": (
            "WIDM 2012 paper on managing and exploiting analysis context across analytical "
            "sessions. Directly relevant to conversational AI and session-aware RAG systems "
            "that maintain dialogue context for improved retrieval."
        ),
    },
    # ── The entries below fill in authors/venue for existing PDF-only corpus
    # entries, sourced from the author's Google Scholar profile (confirmed by
    # the user, not inferred). No abstract/summary is included where the PDF's
    # own text is the actual source of truth for that content — these entries
    # exist to correct authors/venue metadata, not to duplicate/replace the PDF.
    {
        "title": "Discovery of Player Strategies in a Serious Game",
        "authors": ["Hua Li", "Hector Munoz-Avila", "Liana Ke", "Chris Symborski", "Rafael Alonso"],
        "year": 2013,
        "venue": "First AAAI Conference on Human Computation and Crowdsourcing",
        "pdf_filename": "2013-Discovery of Player Strategies in a Serious Game-7636-32518-1-PB.pdf",
    },
    {
        "title": "User Modeling of Skills and Expertise from Resumes",
        "authors": ["Hua Li", "Daniel J. T. Powell", "Mark Clark", "Tifani O'Brien"],
        "year": 2015,
        # Venue given verbatim as "KMIS 2015" on Google Scholar; not expanding the
        # acronym since the full conference name wasn't independently confirmed.
        "venue": "KMIS 2015",
        "pdf_filename": "2015-User Modeling of Skills and Expertise from Resumes-KMIS.pdf",
    },
    {
        "title": "Discovering Virtual Interest Groups across Chat Rooms",
        "authors": ["Hua Li", "Jeff Lau", "Rafael Alonso"],
        "year": 2012,
        "venue": "KMIS 2012",
        "pdf_filename": "2012-Discovering Virtual Interest Groups across Chat Rooms-41315.pdf",
    },
    {
        "title": "Incremental User Modeling with Heterogeneous User Behaviors",
        "authors": ["Rafael Alonso", "Peter Bramsen", "Hua Li"],
        "year": 2010,
        "venue": "KMIS 2010",
        "pdf_filename": "2010-INCREMENTAL USER MODELING WITH HETEROGENEOUS USER BEHAVIORS-30628.pdf",
    },
    {
        "title": "Potential IED Threat System (PITS)",
        "authors": ["Hua Li", "Diane Bramsen", "Rafael Alonso"],
        "year": 2009,
        "venue": "2009 IEEE Conference on Technologies for Homeland Security",
        "pdf_filename": "2009-Potential IED Threat System (PITS)-05168041.pdf",
    },
    {
        # PDF filename in the corpus is "pro-RAMA_cs.pdf" — confirmed by the user to be
        # this same paper under an internal project codename (RAMA), not a separate work.
        "title": "User Modeling for Contextual Suggestion",
        "authors": ["Hua Li", "Rafael Alonso"],
        "year": 2014,
        "venue": "21st Text REtrieval Conference (TREC 2014)",
        "pdf_filename": "pro-RAMA_cs.pdf",
    },
    {
        "title": "Combating Cognitive Biases in Information Retrieval",
        "authors": ["Rafael Alonso", "Hua Li"],
        "year": 2005,
        "venue": "Proc. International Conference on Intelligence Analysis",
        "pdf_filename": "2005 Combating Cognitive Biases in Information Retrieval.pdf",
    },
]
