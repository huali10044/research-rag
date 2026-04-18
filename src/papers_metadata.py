"""
papers_metadata.py — Structured metadata for Hua Li's research papers.

Add your full paper list here. Each entry is used to:
  1. Build a rich text blob for embedding
  2. Populate metadata fields for filtered retrieval
  3. Power the citation/source display in query results

If you have PDFs, drop them in /data/papers/ and they'll be ingested automatically.
The metadata entries complement PDFs by providing clean abstracts and keywords
even when the full PDF text is noisy (scanned pages, headers/footers, etc.).
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
]
