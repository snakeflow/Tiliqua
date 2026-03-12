"""
Core LLM runtime for Tiliqua.

Provides the execution config abstraction and body builders for both
direct API calls and the OpenAI Batch API (Chat Completions and
Responses API variants).

Also provides multi-client utilities (make_clients, MultiClient) for
distributing work across multiple API keys to increase throughput.
"""

import json
import tempfile
import threading


# ---------------------------------------------------------------------------
# Multi-client utilities
# ---------------------------------------------------------------------------

def make_clients(api_keys, model="gpt-5-mini"):
    """
    Create a list of OpenAI clients from a list of API keys.

    Parameters
    ----------
    api_keys : list[str]
        One or more OpenAI API keys.
    model : str
        Default model name (returned for convenience; not stored on clients).

    Returns
    -------
    tuple[list[openai.OpenAI], str]
        (clients, model)

    Example
    -------
    >>> api_keys = ['sk-proj-1', 'sk-proj-2', 'sk-proj-3']
    >>> clients, gpt_model = make_clients(api_keys, model='gpt-5-mini')
    """
    from openai import OpenAI
    clients = [OpenAI(api_key=key) for key in api_keys]
    return clients, model


class MultiClient:
    """
    Thread-safe round-robin wrapper over a list of OpenAI clients.

    Call ``multi.next()`` to get the next client in rotation.
    Useful for distributing parallel requests across multiple API keys.

    Parameters
    ----------
    clients : list[openai.OpenAI]
        Pool of authenticated OpenAI clients.

    Example
    -------
    >>> mc = MultiClient(clients)
    >>> client = mc.next()   # round-robins through the pool
    """

    def __init__(self, clients):
        self._clients = list(clients)
        self._idx = 0
        self._lock = threading.Lock()

    def next(self):
        """Return the next client in round-robin order (thread-safe)."""
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
        return client

    def __len__(self):
        return len(self._clients)

    def __iter__(self):
        return iter(self._clients)

    def __getitem__(self, idx):
        return self._clients[idx]


def default_execution_config(model="gpt-5-mini"):
    """
    Default execution config.
    Compatible with old code.
    """
    return {
        "model": model,
        "tools": None,
        "tool_choice": None,
    }


def build_llm_body(messages, execution_config=None):
    """
    Build the body for OpenAI API calls.
    Supports future tool usage.

    Parameters
    ----------
    messages : list[dict]
        Standard OpenAI messages list.
    execution_config : dict or ExecutionConfig, optional
        Config controlling model and tools. Defaults to default_execution_config().

    Returns
    -------
    dict
        Body ready for client.chat.completions.create(**body).
    """
    if execution_config is None:
        execution_config = default_execution_config()

    def _get(key, default=None):
        if hasattr(execution_config, "get"):
            return execution_config.get(key, default)
        return default

    body = {
        "model": _get("model", "gpt-5-mini"),
        "messages": messages,
    }

    tools = _get("tools")
    if tools:
        body["tools"] = tools

    tool_choice = _get("tool_choice")
    if tool_choice:
        body["tool_choice"] = tool_choice

    return body


def build_batch_job(custom_id, messages, execution_config=None):
    """
    Create a standard OpenAI Batch API job dict.

    Parameters
    ----------
    custom_id : str
        Unique identifier for this job within the batch.
    messages : list[dict]
        Standard OpenAI messages list.
    execution_config : dict or ExecutionConfig, optional

    Returns
    -------
    dict
        A single Batch API request object.
    """
    body = build_llm_body(messages, execution_config)

    job = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }

    return job


def build_responses_body(messages, execution_config=None):
    """
    Build a request body for the OpenAI Responses API.

    Uses the ``input`` key (not ``messages``) as required by /v1/responses.

    Parameters
    ----------
    messages : list[dict]
        Standard role/content message list.
    execution_config : dict or ExecutionConfig, optional

    Returns
    -------
    dict
        Body suitable for use in a Responses API batch job.
    """
    if execution_config is None:
        execution_config = default_execution_config()

    def _get(key, default=None):
        if hasattr(execution_config, "get"):
            return execution_config.get(key, default)
        return default

    body = {
        "model": _get("model", "gpt-4o-mini"),
        "input": messages,
    }
    return body


def build_responses_batch_job(custom_id, messages, execution_config=None):
    """
    Create a Batch API job dict targeting the Responses API endpoint.

    Parameters
    ----------
    custom_id : str
    messages : list[dict]
    execution_config : dict or ExecutionConfig, optional

    Returns
    -------
    dict
        A single Batch API request object for /v1/responses.
    """
    body = build_responses_body(messages, execution_config)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def save_jsonl(lines, suffix):
    """
    Save a list of JSON-serialised strings as a JSONL batch file.

    Parameters
    ----------
    lines : list[str]
        Each element must be a valid JSON string (e.g. json.dumps(job)).
    suffix : str
        File suffix, e.g. "_relevance_chunk001.jsonl".

    Returns
    -------
    str
        Absolute path to the temporary JSONL file.
    """
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        mode="w",
        encoding="utf-8",
    )

    tmp.write("\n".join(lines))
    tmp.close()

    print(f"Batch file saved: {tmp.name}")
    return tmp.name
