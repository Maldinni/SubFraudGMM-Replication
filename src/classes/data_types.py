from dataclasses import dataclass, field
from typing import List


@dataclass
class Embeddings:
    """Conjunto de embeddings de um shard: IDs, textos e vetores alinhados por índice."""
    ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
