---
title: Mobile Banking Biometric Login Specification
doc_id: SPEC-003
department: retail-banking
document_type: product_spec
access_level: public
created_date: 2026-04-02
---
# Mobile Banking Biometric Login Specification

## Behaviour
Customers may enable fingerprint or face login after a one-time password enrolment. Biometric templates never leave the device.

## Fallback
After three failed biometric attempts the app requires the PIN; five PIN failures lock the profile.