"""
João Rodrigues Career & Technical Profile Tools for Bedrock AgentCore
Provides deep, authoritative information on João Rodrigues's background,
experience at Euronext and FanDuel, GenAI implementations, C#/.NET & Python
architecture, and why he is a top candidate for Senior Software Engineer & GenAI roles.
"""

from typing import Any
from strands import tool


CANDIDATE_PROFILE = {
    "name": "João Rodrigues",
    "role": "Senior Software Engineer (MSc) & Generative AI Specialist",
    "email": "joao.g.rodrigues2020@gmail.com",
    "phone": "+351 911 885 628",
    "location": "Zurich, Switzerland (EU Citizen / Portuguese, 30 years old)",
    "education": "Master of Computer Science – Software Engineering (2014 – 2020), Porto University",
    "certifications": [
        "AWS Certified Generative AI Developer – Professional (Ongoing)",
        "Startup Voucher 2020 – Awarded by Portuguese Government (National Innovation Program)",
    ],
    "experience_years": "6+ years",
    "core_specialties": [
        "Generative AI & LLMOps (AWS Bedrock, RAG, Multi-Agent Systems, Fine-Tuning)",
        "Enterprise C# / .NET (DDD, CQRS, Clean Architecture, MediatR)",
        "High-Throughput Python & FastAPI (Async, Microservices, AI Orchestration)",
        "Distributed Systems & Event-Driven Streaming (Apache Kafka, AWS SQS/SNS)",
        "Massive-Scale Resilience (Millions of requests/min at FanDuel for Super Bowl)",
        "Cloud-Native DevOps (AWS, Azure, Docker, Kubernetes, Terraform, CI/CD)",
        "Observability & Reliability (OpenTelemetry, Prometheus, Grafana Loki, 24/7 on-call)",
    ],
}


@tool
def get_candidate_overview() -> str:
    """
    Get an executive summary of João Rodrigues's profile, contact information,
    education, and core value proposition as a Senior Software & GenAI Engineer.
    """
    return (
        "## João Rodrigues — Senior Software Engineer (MSc) & GenAI Specialist\n\n"
        f"- **Location:** {CANDIDATE_PROFILE['location']}\n"
        f"- **Contact:** {CANDIDATE_PROFILE['email']} | {CANDIDATE_PROFILE['phone']}\n"
        f"- **Education:** {CANDIDATE_PROFILE['education']}\n"
        f"- **Experience:** {CANDIDATE_PROFILE['experience_years']} in mission-critical distributed systems and enterprise GenAI.\n\n"
        "### Core Value Proposition:\n"
        "João combines **two highly sought-after engineering pillars**:\n"
        "1. **Rigorous Architectural Discipline for Massive Scale:** Proven at **FanDuel**, architecting and operating platforms handling millions of requests per minute with zero downtime during Tier-1 events like the Super Bowl and March Madness.\n"
        "2. **Hands-On Enterprise GenAI Innovation:** Spearheaded the **first AI project at Euronext Corporate Solutions (iBabs Debrief)**, building multimodal RAG pipelines, fine-tuning open-source LLMs to rival frontier models at lower costs, and deploying real-time speech-to-text and summarization engines."
    )


@tool
def get_candidate_experience(company: str = "") -> str:
    """
    Get detailed professional work experience for João Rodrigues at Euronext, FanDuel,
    Porto University, or startup ventures.

    Args:
        company: Optional filter (e.g., 'Euronext', 'FanDuel', 'Porto', or empty for all).
    """
    comp_lower = (company or "").lower()

    euronext_info = (
        "### 🏛️ Euronext Corporate Solutions (Feb 2023 – Present) — Software Developer\n"
        "**Key Focus:** Enterprise AI & LLMs, Video Conferencing, Capital Markets Infrastructure\n"
        "- **First AI Project at Euronext (iBabs Debrief):** Architected the backend from the ground up using C#, .NET, and Python. Integrated AI models for real-time speech-to-text, meeting summarization, live subtitling, and intelligent document processing.\n"
        "- **Advanced Multimodal RAG:** Built end-to-end RAG pipelines ingesting unstructured audio and documents to act as domain-specific enterprise experts.\n"
        "- **LLM Fine-Tuning & Cost Optimization:** Fine-tuned open-source models to match GPT-4 / Claude performance on specialized tasks, significantly reducing inference costs.\n"
        "- **Enterprise Video Platform:** Built a video conferencing platform from scratch (similar to Microsoft Teams), utilizing DDD, CQRS (MediatR), Clean Architecture, and Strategy/Factory patterns.\n"
        "- **Real-Time Notification Engine:** Built a high-throughput market update notification service in Golang on AWS (SES, SNS, SQS, ECR, ECS).\n"
        "- **Microservices Orchestration:** Kubernetes, Redis caching, Keycloak auth, OpenTelemetry, Prometheus, and Grafana Loki dashboards."
    )

    fanduel_info = (
        "### 🏈 FanDuel (Oct 2020 – Feb 2023) — Software Developer\n"
        "**Key Focus:** Extreme-Scale Distributed Systems, Real-Time ETL, Mission-Critical High Availability\n"
        "- **Massive Scale Operations:** Scaled platforms across all U.S. states to handle **millions of requests per minute** during the **Super Bowl** and **March Madness** with sub-second latency and zero downtime.\n"
        "- **Real-Time Streaming ETL (Apache Kafka):** Engineered streaming pipelines processing millions of messages per minute under extreme peak traffic.\n"
        "- **Core Product Innovation (Same Game Parlay):** Built foundational products including the 'Same Game Parlay' (SGP), which became a multi-million-dollar commercial success and industry-defining revenue driver.\n"
        "- **Analytics & Reporting Engine:** High-throughput analytics platform tracking user betting volumes, margins, and performance using C#, Python, Kafka, and PostgreSQL.\n"
        "- **24/7 Production On-Call:** Responsible for tier-1 user-facing services (Bet Tracker, Cashout), ensuring 99.99%+ reliability."
    )

    other_info = (
        "### 🎓 Porto University & Entrepreneurship\n"
        "- **Master's Thesis (Feb 2020 – Oct 2020):** Transitioned BCCT.core into a Python/Django platform integrating deep learning models for breast cancer treatment outcome analysis.\n"
        "- **Startup Voucher 2020 (Government Innovation Award):** Founded and built a travel-tech startup from the ground up using Python, Django, and PostgreSQL, driving both technology and business strategy."
    )

    if "euronext" in comp_lower:
        return euronext_info
    elif "fanduel" in comp_lower:
        return fanduel_info
    elif "porto" in comp_lower or "startup" in comp_lower:
        return other_info

    return f"{euronext_info}\n\n{fanduel_info}\n\n{other_info}"


@tool
def get_genai_and_technical_skills(domain: str = "") -> str:
    """
    Get João Rodrigues's technical stack, split into GenAI, Backend Architecture,
    Distributed Systems, Cloud & DevOps, or Databases.

    Args:
        domain: Optional filter (e.g., 'GenAI', 'Backend', 'Cloud', 'Architecture', or empty for full stack).
    """
    d_lower = (domain or "").lower()

    genai_section = (
        "### 🤖 Generative AI & LLMOps:\n"
        "- **AWS Bedrock Ecosystem:** AgentCore, Strands SDK, AgentSquad, Amazon Titan v2 embeddings, Nova Micro/Lite.\n"
        "- **Multi-Agent Systems:** Autonomous supervisor routing, specialized agent handoffs, session isolation.\n"
        "- **Advanced RAG Pipelines:** Serverless vector stores, semantic & context-enriched chunking, hybrid keyword/vector retrieval.\n"
        "- **Model Fine-Tuning:** Fine-tuning open-source LLMs to match frontier model accuracy at lower cost.\n"
        "- **Speech & Multimodal:** Real-time speech-to-text, live subtitling, meeting summarization, audio ingestion.\n"
        "- **Evaluation & Guardrails:** Automated LLM-as-a-Judge benchmarks, PII masking, and prompt-injection defense."
    )

    backend_section = (
        "### ⚡ Backend Engineering & Distributed Systems:\n"
        "- **Languages:** C#, .NET 8/9, Python (FastAPI, Django), Golang, Java, SQL.\n"
        "- **Architectural Patterns:** Domain-Driven Design (DDD), Clean Architecture, CQRS (MediatR), Repository, Strategy, and Factory patterns.\n"
        "- **Event-Driven Messaging:** Apache Kafka (millions of msgs/min), AWS SQS, SNS, EventBridge.\n"
        "- **Performance & Scale:** Highly concurrent, memory-efficient, fault-tolerant architectures designed for extreme traffic spikes."
    )

    cloud_section = (
        "### ☁️ Cloud, DevOps & Observability:\n"
        "- **Cloud Providers:** AWS (Bedrock, ECS, ECR, S3, Lambda, SQS, SNS), Azure (Blob Storage, App Services).\n"
        "- **Containers & Orchestration:** Docker, Kubernetes, Docker Compose, Terraform.\n"
        "- **Observability & Monitoring:** OpenTelemetry, Prometheus, Grafana Loki, CloudWatch, structured logging (Serilog)."
    )

    if "genai" in d_lower or "ai" in d_lower or "llm" in d_lower:
        return genai_section
    elif "backend" in d_lower or "c#" in d_lower or "net" in d_lower or "python" in d_lower:
        return backend_section
    elif "cloud" in d_lower or "devops" in d_lower or "aws" in d_lower:
        return cloud_section

    return f"{genai_section}\n\n{backend_section}\n\n{cloud_section}"


@tool
def get_candidate_pitch_and_why_hire() -> str:
    """
    Get João Rodrigues's personal pitch, cover letter highlights, and why he is an
    exceptional hire for Senior Software Engineering and GenAI roles.
    """
    return (
        "## Why Hire João Rodrigues?\n\n"
        "### 1. Dual Mastery: Heavy Distributed Systems + Modern GenAI\n"
        "Most engineers either specialize in classical backend infrastructure OR experiment with GenAI wrappers. "
        "João has **battle-tested experience in both**:\n"
        "- He knows how to keep distributed systems alive when millions of users hit the platform during the **Super Bowl** at FanDuel.\n"
        "- He knows how to architect enterprise-ready **AI products from scratch** at Euronext with strict latency, privacy, and cost constraints.\n\n"
        "### 2. Product-Minded & Customer-Centric\n"
        "João does not just write code in isolation; he takes full ownership of product outcomes. At Euronext, he transitioned "
        "from startup agility to enterprise corporate standards, working directly to understand customer needs and delivering "
        "tools like iBabs Debrief that achieved widespread adoption.\n\n"
        "### 3. Engineering Rigor & Clean Code Advocate\n"
        "Deep expertise in Clean Architecture, DDD, CQRS, automated testing (TDD), zero-downtime deployments, and end-to-end observability "
        "(OpenTelemetry, Prometheus). He writes code built for maintainability, fault-tolerance, and scale.\n\n"
        "### 4. Availability & Relocation\n"
        "Based in **Zurich, Switzerland** (EU Citizen), available for high-impact Senior Software Engineer and GenAI Engineer roles."
    )
