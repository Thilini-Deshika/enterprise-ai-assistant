"""Typed application errors. Each maps to an HTTP status and a stable machine-readable code."""
from __future__ import annotations


class AppError(Exception):
    code = "internal_error"
    http_status = 500
    user_message = "Something went wrong. Please try again."

    def __init__(self, message: str | None = None, **details):
        super().__init__(message or self.user_message)
        self.message = message or self.user_message
        self.details = details


class ValidationFailed(AppError):
    code, http_status = "validation_failed", 422


class AuthenticationError(AppError):
    code, http_status = "unauthenticated", 401


class AuthorizationError(AppError):
    code, http_status = "forbidden", 403


class RateLimitExceeded(AppError):
    code, http_status = "rate_limited", 429


class PromptInjectionBlocked(AppError):
    code, http_status = "prompt_injection_blocked", 400


class GuardrailViolation(AppError):
    code, http_status = "guardrail_violation", 422


class DependencyUnavailable(AppError):
    code, http_status = "dependency_unavailable", 503


class LLMUnavailable(DependencyUnavailable):
    code = "llm_unavailable"


class VectorStoreUnavailable(DependencyUnavailable):
    code = "vector_store_unavailable"


class MCPUnavailable(DependencyUnavailable):
    code = "mcp_unavailable"


class ToolError(AppError):
    code, http_status = "tool_error", 502


class ToolTimeout(ToolError):
    code = "tool_timeout"


class ApprovalRequired(AppError):
    """Raised by the tool registry when a tool needs human approval before it can run."""
    code, http_status = "approval_required", 409


class BudgetExceeded(AppError):
    """RLM research budget (searches / LLM calls / depth / time) exhausted."""
    code, http_status = "budget_exceeded", 429
