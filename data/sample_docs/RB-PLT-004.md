---
title: "Runbook: Kafka Consumer Lag Remediation"
doc_id: RB-PLT-004
department: platform-engineering
document_type: runbook
access_level: internal
created_date: 2026-04-12
---
# Runbook: Kafka Consumer Lag Remediation

## Detection
Consumer lag above 5 minutes on settlement topics pages the platform on-call.

## Remediation
Scale the consumer group (maximum equals partition count), check for a partition rebalance storm, and pause non-critical consumers. Use the replay tool to reprocess events after recovery.

## Prevention
Autoscale on lag and review partition counts every quarter.