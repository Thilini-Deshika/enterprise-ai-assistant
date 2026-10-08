---
title: "Runbook: Card Processor Timeout Failover"
doc_id: RB-PAY-002
department: payments
document_type: runbook
access_level: internal
created_date: 2026-02-10
---
# Runbook: Card Processor Timeout Failover

## Detection
Processor timeout rate above 10% for 3 minutes (alert CARDPROC-TIMEOUT).

## Failover Procedure
Switch the routing flag processor.primary to the secondary processor via the routing console (two approvals). Confirm authorisation success above 97% within 5 minutes.

## Fail-back
Fail back only after the primary reports healthy for 30 minutes and during a low-traffic window.