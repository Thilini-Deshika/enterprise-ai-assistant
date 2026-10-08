---
title: "Runbook: TLS Certificate Renewal and Expiry Response"
doc_id: RB-PLT-003
department: platform-engineering
document_type: runbook
access_level: internal
created_date: 2026-03-05
---
# Runbook: TLS Certificate Renewal and Expiry Response

## Inventory
All certificates are tracked in the certificate inventory with owner and expiry date. Alerts fire at 21, 7 and 1 days before expiry.

## Emergency Renewal
Generate a CSR from the vault, request issuance from the internal CA, deploy through the standard pipeline, and verify the mutual-TLS handshake with the synthetic check.

## Verification
Confirm zero handshake errors for 15 minutes and update the inventory entry.