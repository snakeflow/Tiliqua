"""
Batch runtime for Tiliqua.

Wraps the OpenAI Batch API: submit, poll status, and download results.
The OpenAI client is injected by the caller, keeping this module testable.
"""

import os
import json


def submit_batch(client, jsonl_path, endpoint="/v1/chat/completions", completion_window="24h"):
    """
    Upload a JSONL file and create a batch job on OpenAI.

    Parameters
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client.
    jsonl_path : str
        Local path to the .jsonl batch file.
    endpoint : str
        OpenAI endpoint for the batch (default: /v1/chat/completions).
    completion_window : str
        Maximum duration for the batch (default: "24h").

    Returns
    -------
    str
        The batch_id assigned by OpenAI.
    """
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window=completion_window,
    )

    print(f"Batch submitted: {batch.id}")
    return batch.id


def retrieve_batch_result(client, batch_id):
    """
    Poll OpenAI for the current status of a batch job.

    Parameters
    ----------
    client : openai.OpenAI
    batch_id : str

    Returns
    -------
    tuple[str, str | None]
        (status, output_file_id). output_file_id is None if not yet complete.
        Possible statuses: "validating", "in_progress", "completed",
        "failed", "expired", "cancelling", "cancelled".
    """
    batch_obj = client.batches.retrieve(batch_id)

    status = getattr(batch_obj, "status", "unknown")
    output_file_id = getattr(batch_obj, "output_file_id", None)

    return status, output_file_id


def submit_batches_multi_client(
    clients,
    jsonl_paths,
    endpoint="/v1/chat/completions",
    completion_window="24h",
):
    """
    Distribute multiple JSONL batch files across a pool of OpenAI clients.

    Batch files are assigned to clients in round-robin order so that
    concurrent batches run under different API keys, avoiding per-key
    rate limits.

    Parameters
    ----------
    clients : list[openai.OpenAI] or MultiClient
        Pool of authenticated clients.
    jsonl_paths : list[str]
        Local paths to the JSONL batch files to submit.
    endpoint : str
        OpenAI batch endpoint (default: /v1/chat/completions).
    completion_window : str

    Returns
    -------
    list[tuple[int, str]]
        List of (client_index, batch_id) pairs, one per submitted file.

    Example
    -------
    >>> results = submit_batches_multi_client(clients, [ris_jsonl, stmt_jsonl],
    ...                                       endpoint="/v1/responses")
    >>> for client_idx, batch_id in results:
    ...     print(client_idx, batch_id)
    """
    results = []
    n = len(clients)
    for i, jsonl_path in enumerate(jsonl_paths):
        client = clients[i % n]
        batch_id = submit_batch(client, jsonl_path, endpoint=endpoint, completion_window=completion_window)
        results.append((i % n, batch_id))
    return results


def download_batch_output(client, output_file_id, save_path=None):
    """
    Download and parse the JSONL output of a completed batch job.

    Parameters
    ----------
    client : openai.OpenAI
    output_file_id : str
        The output_file_id returned by retrieve_batch_result().
    save_path : str, optional
        If provided, raw JSONL is also written to this path.

    Returns
    -------
    list[dict]
        One dict per completed request, parsed from JSONL.
    """
    file_content = client.files.content(output_file_id).text

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(file_content)

    lines = [
        json.loads(line)
        for line in file_content.strip().splitlines()
        if line.strip()
    ]

    return lines
