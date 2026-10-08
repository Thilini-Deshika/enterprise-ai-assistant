---
title: Security Zoning and Key Management
doc_id: ARCH-003
department: security
document_type: architecture
access_level: restricted
created_date: 2026-01-05
---
# Security Zoning and Key Management

## Network Zones
Zone A (internet-facing), Zone B (application), Zone C (data), Zone D (HSM and key services). Only Zone B may call Zone D through the key broker.

## Key Hierarchy
Master keys live in HSMs with split-knowledge custodians. Application keys are wrapped and rotated every 90 days.