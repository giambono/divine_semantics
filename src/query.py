
import concurrent.futures
from typing import Any, Union, List
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

import config


def evaluate_query(
        client: Any,
        collection_name: str,
        query_text: str,
        expected_index: int,
        model: Any,
        author_ids: Union[int, List[int]]=None,
        type_ids: Union[int, List[int]] = 1,
        model_payload_key: str = None
) -> bool:
    """
    Evaluate a text query against a Qdrant collection and check for the presence of an expected index.

    This function encodes the provided query text using the given model, constructs a search filter
    using the provided author and type IDs, and then performs a search in the specified Qdrant collection.
    It checks if the expected index is present in the 'cumulative_indices' field of the top search hit.

    Args:
        client: The Qdrant client used to execute the search.
        collection_name (str): The name of the collection in Qdrant.
        query_text (str): The query text to be encoded and searched.
        expected_index (int): The index to look for in the top hit's 'cumulative_indices'.
        model: A model instance with an 'encode' method to generate a query embedding.
        author_ids (int or list): A single author ID or a list of author IDs used for filtering the search.
        type_ids (int or list): A single type ID or a list of type IDs used for filtering the search.

    Returns:
        bool: True if the expected_index is found in the 'cumulative_indices' of the top search result;
              False if no matching verse is found or the expected_index is not present.
    """

    author_ids = [author_ids] if isinstance(author_ids, int) else author_ids
    type_ids = [type_ids] if isinstance(type_ids, int) else type_ids

    # Compute the query embedding from your query_text
    query_embedding = model.encode(query_text)

    model_identifier = model.model_card_data.base_model
    if model_payload_key is None:
        model_payload_key = next((k for k, v in config.MODELS.items() if v == model_identifier), None)
    if model_payload_key is None:
        raise ValueError(f"model {model_identifier} is not in collection {collection_name}")

    # Build the base conditions that always apply.
    must_conditions = [
        FieldCondition(key="type_id", match=MatchAny(any=type_ids)),
        FieldCondition(key="model", match=MatchAny(any=[model_payload_key]))
    ]

    # Only add the author filter if author_id is not None.
    if author_ids is not None:
        must_conditions.insert(1, FieldCondition(key="author_id", match=MatchAny(any=author_ids)))

    search_filter = Filter(must=must_conditions)

    # Perform the search in Qdrant
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=1,  # we only care about the top match
        query_filter=search_filter
    )

    if hits:
        # check if the expected index is included in tercet's cumulative indices
        top_hit = hits[0]
        return expected_index in top_hit.payload['cumulative_indices']

    else:
        print("No matching verses found.")
        return False


def process_query(row, client, collection_name, model, author_ids, type_ids, model_payload_key):
    query_text = row["transformed_text"]
    expected_index = row["expected_index"]
    result = evaluate_query(client, collection_name, query_text, expected_index, model, author_ids, type_ids, model_payload_key)
    return query_text, result


def run_evaluation(qdrant_client, collection_name, model, author_ids, type_ids, test_queries, model_payload_key):

    # Use a ThreadPoolExecutor for concurrent evaluation of queries (I/O-bound)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_query, row, qdrant_client, collection_name, model, author_ids, type_ids, model_payload_key)
            for _, row in test_queries.iterrows()
        ]
        out_collect = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Calculate performance
    true_count = sum(flag for _, flag in out_collect)
    performance = true_count / len(out_collect)

    return out_collect, performance
