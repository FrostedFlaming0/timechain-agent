"""
agent — minimal agent loop showing how chain + retrieval plug into an LLM.

The model is abstracted behind a simple interface so you can swap in
Claude, GPT-4, Llama, or anything else. This file shows the flow:

    1. User input arrives -> append as 'observation' record.
    2. Retrieve relevant prior context from the chain.
    3. Build a prompt with retrieved context + user input.
    4. Call the model.
    5. Append model output as 'response' record, with refs to retrieved records.
    6. Optionally seal a Merkle batch and (in production) anchor the root.

The chain is doing real work here: every action and observation is committed,
referenceable, signed, and tamper-evident. Retrieval pulls structured prior
context. The model is stateless across turns; the chain is the state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from chain import Chain, Record
from retrieval import Retriever, RetrievalHit


# Pluggable LLM interface — implement this for whatever model you're using.
LLMCall = Callable[[str], str]


@dataclass
class AgentTurn:
    observation_record: Record
    retrieved: list[Record]
    response_record: Record
    response_text: str


class Agent:
    def __init__(self, chain: Chain, retriever: Retriever, llm: LLMCall):
        self.chain = chain
        self.retriever = retriever
        self.llm = llm

    def commit_genesis(self, founding_commitments: list[str]) -> Record:
        """
        Write the agent's founding commitments as record 0. These become the
        anchor for drift detection. Cannot be modified without breaking the chain.
        """
        if self.chain.length() > 0:
            raise RuntimeError("genesis already committed")
        return self.chain.append(
            "genesis",
            {
                "commitments": founding_commitments,
                "schema_version": 1,
            },
        )

    def turn(self, user_input: str, retrieve_k: int = 5) -> AgentTurn:
        # 1. Commit observation
        obs = self.chain.append("observation", {"text": user_input})

        # 2. Retrieve relevant context
        context = self.retriever.build_context(
            query=user_input, k_semantic=retrieve_k, n_recent=3
        )

        # 3. Build prompt
        prompt = self._format_prompt(user_input, context)

        # 4. Call model
        response_text = self.llm(prompt)

        # 5. Commit response with refs to what informed it
        refs = [r.record_hash for r in context] + [obs.record_hash]
        response = self.chain.append(
            "response", {"text": response_text}, refs=refs
        )

        return AgentTurn(
            observation_record=obs,
            retrieved=context,
            response_record=response,
            response_text=response_text,
        )

    def _format_prompt(self, user_input: str, context: list[Record]) -> str:
        ctx_blocks = []
        for rec in context:
            try:
                content_repr = json.dumps(rec.content, ensure_ascii=False)
            except (TypeError, ValueError):
                content_repr = str(rec.content)
            ctx_blocks.append(
                f"[record {rec.index} | type={rec.type}] {content_repr}"
            )
        ctx = "\n".join(ctx_blocks) if ctx_blocks else "(no prior context)"
        return (
            "You are an agent with persistent memory in a hash-chained log.\n"
            "Retrieved prior records:\n"
            f"{ctx}\n\n"
            f"Current input: {user_input}\n\n"
            "Respond. Be consistent with prior records."
        )


# ---------------------------------------------------------------------------
# Mock LLM for the demo (so the demo is reproducible and offline)
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Deterministic fake model. Echoes back a structured summary of context +
    input so the demo shows the chain doing its job without needing API keys.
    Replace with a real LLM client.
    """

    def __call__(self, prompt: str) -> str:
        # Pull just the current input line for the echo
        lines = prompt.splitlines()
        current = next((l for l in lines if l.startswith("Current input:")), "")
        return f"Acknowledged. {current.replace('Current input: ', '')}"
