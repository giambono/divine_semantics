import pytest
import numpy as np
import pandas as pd
import json
import uuid

import sys
import os


from src.compute_embeddings import weighted_avg_embedding_qdrant

# --- Fake classes to simulate Qdrant behavior ---

class FakePoint:
    def __init__(self, payload, vector):
        self.payload = payload
        self.vector = vector

class FakePointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload

class FakeQdrantClient:
    def __init__(self, points):
        self.points = points  # list of FakePoint objects
        self.upserted = []    # records points that have been upserted

    def scroll(self, collection_name, limit, offset, with_vectors, with_payload):
        # For simplicity, return all points in one batch on the first call
        if offset is None:
            return self.points, None
        else:
            return [], None

    def upsert(self, collection_name, points):
        self.upserted.extend(points)

# --- Pytest fixture to override the PointStruct used in the function ---
@pytest.fixture(autouse=True)
def override_pointstruct(monkeypatch):
    # Override the models.PointStruct in the compute_embeddings module with our fake.
    from database.qdrant import compute_embeddings
    monkeypatch.setattr(compute_embeddings.models, "PointStruct", FakePointStruct)

# --- Test cases ---

def test_no_points_found():
    """
    When no points are returned from scroll, the function should raise an Exception.
    """
    fake_client = FakeQdrantClient(points=[])
    with pytest.raises(Exception, match="No points found in collection"):
        weighted_avg_embedding_qdrant(
            model_name="test_model",
            collection_name="test_collection",
            qdrant_client=fake_client,
            author_weights={1: 0.5},
            batch_size=1000,
            type_id=1,
            store_in_qdrant=False
        )

def test_existing_weighted_embedding():
    """
    If a point already exists in the collection with the weighted model and matching
    author weights, the function should raise a ValueError.
    """
    global_author_weights = {1: 0.5, 2: 0.5}
    # Serialize weights to ensure consistency.
    sorted_author_weights = json.dumps(dict(sorted(global_author_weights.items())))
    weighted_model = "weighted_embedding_test_model"
    # Create a point that simulates an existing weighted embedding.
    existing_payload = {
        "model": weighted_model,
        "applied_author_weights": sorted_author_weights
    }
    fake_point = FakePoint(payload=existing_payload, vector=[0, 0, 0])
    fake_client = FakeQdrantClient(points=[fake_point])

    with pytest.raises(ValueError, match="Weighted average embedding with the given author weights already exists."):
        weighted_avg_embedding_qdrant(
            model_name="test_model",
            collection_name="test_collection",
            qdrant_client=fake_client,
            author_weights=global_author_weights,
            batch_size=1000,
            type_id=1,
            store_in_qdrant=False
        )

def test_compute_weighted_embedding():
    """
    Tests that the function correctly computes the weighted average for points in the same group.
    """
    author_weights = {1: 0.5, 2: 0.5}
    # Create two points that belong to the same verse segment.
    payload1 = {
        "model": "test_model",
        "type_id": 1,
        "cantica_id": 1,
        "canto": 1,
        "start_verse": 1,
        "end_verse": 1,
        "author_id": 1
    }
    payload2 = {
        "model": "test_model",
        "type_id": 1,
        "cantica_id": 1,
        "canto": 1,
        "start_verse": 1,
        "end_verse": 1,
        "author_id": 2
    }
    vector1 = [1, 2, 3]
    vector2 = [3, 2, 1]
    fake_point1 = FakePoint(payload=payload1, vector=vector1)
    fake_point2 = FakePoint(payload=payload2, vector=vector2)
    fake_client = FakeQdrantClient(points=[fake_point1, fake_point2])

    df = weighted_avg_embedding_qdrant(
        model_name="test_model",
        collection_name="test_collection",
        qdrant_client=fake_client,
        author_weights=author_weights,
        batch_size=1000,
        type_id=1,
        store_in_qdrant=False
    )

    # There should be one group with one weighted average embedding.
    assert not df.empty
    weighted_col = "weighted_embedding_test_model"
    # Compute the expected weighted average using equal weights.
    expected = np.average(np.array([vector1, vector2]), axis=0, weights=[0.5, 0.5])
    computed = np.array(df.iloc[0][weighted_col])
    np.testing.assert_allclose(computed, expected)

def test_store_in_qdrant():
    """
    When store_in_qdrant is True, the function should call upsert with the computed embeddings.
    """
    author_weights = {1: 1.0}
    payload = {
        "model": "test_model",
        "type_id": 1,
        "cantica_id": 1,
        "canto": 1,
        "start_verse": 1,
        "end_verse": 1,
        "author_id": 1
    }
    vector = [1, 1, 1]
    fake_point = FakePoint(payload=payload, vector=vector)
    fake_client = FakeQdrantClient(points=[fake_point])

    df = weighted_avg_embedding_qdrant(
        model_name="test_model",
        collection_name="test_collection",
        qdrant_client=fake_client,
        author_weights=author_weights,
        batch_size=1000,
        type_id=1,
        store_in_qdrant=True,
        upsert_batch_size=1  # Trigger upsert immediately for testing
    )

    # Verify that upsert was called (i.e. at least one point was upserted)
    assert len(fake_client.upserted) > 0
    for point in fake_client.upserted:
        assert point.payload.get("model") == "weighted_embedding_test_model"
