"""Domain vocabulary shared by ingestion, filters, heuristics and the MCP server."""
DEPARTMENTS = ("payments", "platform-engineering", "security", "customer-support", "risk-compliance",
               "retail-banking", "data-engineering")
DOC_TYPES = ("policy", "architecture", "runbook", "incident", "product_spec", "meeting_notes")
SERVICES = ("payment-gateway", "card-processing", "ledger-service", "fraud-engine", "mobile-banking-api",
            "notification-service", "customer-profile", "kafka-platform")
SEVERITIES = ("sev1", "sev2", "sev3", "sev4")
ROOT_CAUSE_CATEGORIES = (
    "database connection pool exhaustion", "expired TLS certificate", "third-party processor timeout",
    "faulty configuration deployment", "message queue consumer lag", "fraud rule misconfiguration",
)
