---
title: Payment Operations Change Policy
doc_id: POL-PAY-002
department: payments
document_type: policy
access_level: internal
created_date: 2025-08-15
---
# Payment Operations Change Policy

## Change Freeze
No production changes to payment systems between 25 November and 2 January, or during month-end settlement (last two business days).
Emergency changes require the Head of Payments and an incident commander.

## Four-Eyes Principle
Every configuration or rule change affecting payment routing, limits or fraud thresholds requires peer review and a second approver.

## Canary and Rollback
Routing and fraud-rule changes must be released to a canary (5% of traffic) for 30 minutes with automated rollback on error-rate regressions.