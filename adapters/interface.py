# adapters/interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class AdapterBase(ABC):
    """
    Adapter interface: implement for each extraction backend.
    """

    name: str = "base_adapter"

    def __init__(self, options: Dict[str, Any] = None):
        self.options = options or {}

    @abstractmethod
    def probe(self) -> Dict[str, Any]:
        """Return info about connected device(s) and capabilities."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, artifact_list: List[str], out_dir: str, metadata: Dict[str,Any]) -> Dict[str,Any]:
        """
        Perform extraction for requested artifacts.
        - artifact_list: list of artifact keys (e.g., ['contacts','sms','files'])
        - out_dir: destination directory; must create files and .meta.json
        - metadata: runtime metadata (must include consent_id)
        Returns: dict with summary and produced artifact paths.
        """
        raise NotImplementedError

    @abstractmethod
    def status(self) -> Dict[str,Any]:
        """Return running status / health."""
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None:
        """Attempt to abort current extraction."""
        raise NotImplementedError
