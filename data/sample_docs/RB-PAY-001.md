---
title: "Runbook: Payment Gateway Degradation"
doc_id: RB-PAY-001
department: payments
document_type: runbook
access_level: internal
created_date: 2026-01-20
---
# Runbook: Payment Gateway Degradation

## Symptoms
Authorisation latency above 2 seconds, error rate above 5%, or alerts PAYGW-ERR-HIGH / PAYGW-POOL-SAT.

## Triage Steps
1. Check the connection pool dashboard for saturation above 85%. 2. Check upstream processor health. 3. Review recent deployments in the last 2 hours.

## Mitigation
If the pool is saturated, enable the circuit breaker flag and scale gateway pods to 12. If a recent deploy is suspected, roll back immediately. Page the payments on-call if not recovered in 15 minutes.

## Escalation
Sev1 conditions go to the Incident Commander on-call and the Head of Payments.