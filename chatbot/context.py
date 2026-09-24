"""Session context state store for findings chatbot."""

from __future__ import annotations

from typing import Any, Dict, Optional


class SessionContextManager:
    """In-memory session state manager storing active analysis results per session ID."""

    def __init__(self) -> None:
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, list[Dict[str, str]]] = {}

    def set_analysis_result(self, session_id: str, analysis_result: Dict[str, Any]) -> None:
        """Store the latest analysis result for a session."""
        self._contexts[session_id] = analysis_result

    def get_analysis_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored analysis result for a session."""
        return self._contexts.get(session_id)

    def add_message(self, session_id: str, role: str, message: str) -> None:
        """Append message to conversation history."""
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append({"role": role, "content": message})

    def get_history(self, session_id: str) -> list[Dict[str, str]]:
        """Retrieve conversation history for session."""
        return self._history.get(session_id, [])

    def clear(self, session_id: str) -> None:
        """Clear context for session."""
        self._contexts.pop(session_id, None)
        self._history.pop(session_id, None)


# Global instance
session_manager = SessionContextManager()
