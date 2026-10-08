---
title: Event Streaming Architecture (kafka-platform)
doc_id: ARCH-002
department: platform-engineering
document_type: architecture
access_level: internal
created_date: 2025-12-08
---
# Event Streaming Architecture

## Topology
Three-broker clusters per region with rack awareness; topics are partitioned by account identifier. Consumer groups scale up to the partition count.

## Delivery Guarantees
At-least-once delivery with idempotent consumers; settlement topics retain 7 days to allow replay.

## Operations
Lag, rebalance rate and under-replicated partitions are the primary health signals.