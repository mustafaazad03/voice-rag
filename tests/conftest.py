from __future__ import annotations

import pytest

from vrag import config
from vrag.config import Settings
from vrag.models import Document

SAMPLE = [
    (
        "A corporation is a company or group of people authorised to act as a single entity "
        "and recognised as such in law. Early incorporated entities were established by charter. "
        "Most jurisdictions now allow the creation of new corporations through registration."
    ),
    (
        "A company is incorporated in a specific nation, often within a smaller subset such as a "
        "state or province. The corporation is then governed by the laws of incorporation in that "
        "state. A corporation may issue stock, either private or public."
    ),
    (
        "Photosynthesis is the process used by plants to convert light energy into chemical energy. "
        "The process releases oxygen as a by-product. Chlorophyll absorbs light most strongly in the "
        "blue and red parts of the spectrum."
    ),
    (
        "The boiling point of water at sea level is 100 degrees Celsius. Boiling point decreases "
        "with altitude because atmospheric pressure falls. At 2000 metres water boils near 93 degrees."
    ),
    (
        "कॉर्पोरेशन एक कंपनी या लोगों का समूह है जिसे कानून में एकल इकाई के रूप में कार्य करने के लिए अधिकृत किया गया है। "
        "अधिकांश देशों में अब पंजीकरण के माध्यम से नई कंपनियां बनाई जा सकती हैं।"
    ),
]


@pytest.fixture(scope="session")
def settings():
    # _env_file=None: a developer's local .env must not reach the tests. With real
    # STT keys loaded the provider tests stop being hermetic and call the live API.
    return Settings(_env_file=None, offload_cpu=False)


@pytest.fixture(scope="session", autouse=True)
def _global_settings(settings):
    """Make get_settings() return the isolated object too.

    Code under test reaches for the global singleton (the rate-limit middleware,
    for one), which would otherwise load .env behind the fixtures' back.
    """
    config._settings = settings
    yield
    config.reset_settings()


@pytest.fixture(scope="session")
def documents() -> list[Document]:
    return [
        Document(
            doc_id=f"d{i}",
            text=text,
            lang="hin" if i == 4 else "eng",
            query_id=100 + i,
            query_type="DESCRIPTION",
            is_selected=i in (0, 4),
            trust=0.85 if i in (0, 4) else 0.5,
        )
        for i, text in enumerate(SAMPLE)
    ]


@pytest.fixture(scope="session")
def embedder():
    from vrag.index.embedder import get_embedder

    return get_embedder()


@pytest.fixture(scope="session")
def store(documents, embedder, settings):
    from vrag.index.store import ChunkStore

    return ChunkStore.build(
        documents, strategy="hierarchical", settings=settings, embedder=embedder
    )


@pytest.fixture()
def pipeline(store, embedder, settings):
    from vrag.cache import ResponseCache
    from vrag.harness.pipeline import RAGPipeline
    from vrag.stt import STTRouter

    return RAGPipeline(
        store,
        settings=settings,
        embedder=embedder,
        cache=ResponseCache(settings),
        stt=STTRouter(settings),
    )
