"""Role-based access control. The single source of truth for "who may do / see what".

Two independent enforcement axes:
  1. Permissions  -> which TOOLS a role may invoke (checked inside ToolRegistry, never by the LLM).
  2. Access levels -> which DOCUMENTS a role may read (enforced as Pinecone namespaces + metadata filter).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "administrator"


class Permission(str, Enum):
    CHAT = "chat"
    SEARCH = "search"
    ANALYTICS = "tool:analytics"
    MCP_READ = "tool:mcp_read"
    MCP_WRITE = "tool:mcp_write"
    ADMIN_TOOLS = "tool:admin"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({Permission.CHAT, Permission.SEARCH}),
    Role.ANALYST: frozenset({Permission.CHAT, Permission.SEARCH, Permission.ANALYTICS, Permission.MCP_READ}),
    Role.ADMIN: frozenset(Permission),
}

# Document sensitivity tiers. Each tier is a Pinecone *namespace*, so a viewer's query never even
# touches the namespaces they cannot read (isolation by construction, not by post-filtering).
ACCESS_LEVELS = ("public", "internal", "confidential", "restricted")
ROLE_ACCESS_LEVELS: dict[Role, tuple[str, ...]] = {
    Role.VIEWER: ("public", "internal"),
    Role.ANALYST: ("public", "internal", "confidential"),
    Role.ADMIN: ACCESS_LEVELS,
}


@dataclass(frozen=True)
class UserContext:
    user_id: str
    name: str
    role: Role
    department: str

    def has(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]

    @property
    def allowed_access_levels(self) -> tuple[str, ...]:
        return ROLE_ACCESS_LEVELS[self.role]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UserContext":
        return cls(d["user_id"], d["name"], Role(d["role"]), d["department"])
