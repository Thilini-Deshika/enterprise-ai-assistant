---
title: Payments Platform Architecture
doc_id: ARCH-001
department: payments
document_type: architecture
access_level: internal
created_date: 2025-11-03
---
# Payments Platform Architecture

## Overview
The payment-gateway is a stateless service fronting card-processing and ledger-service. Requests flow API gateway, payment-gateway, fraud-engine (synchronous scoring), then card-processing to the external processor, and finally the ledger-service.

## Resilience
Gateway uses a bounded database connection pool, circuit breakers per downstream, and a secondary processor for failover. Settlement events are published to kafka-platform.

## Data
The ledger is the system of record; payment-gateway keeps only transient state in a managed Postgres cluster.