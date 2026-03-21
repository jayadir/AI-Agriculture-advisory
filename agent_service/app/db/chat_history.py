"""
Helpers for reading and writing the `chat_history` MongoDB collection.

Each user has ONE document keyed by phone_number. Turns are appended in
chronological order; we always read only the last N turns for context.
"""
from datetime import datetime, timezone
from typing import List, Dict

COLLECTION = "chat_history"
MAX_TURNS = 200  # Hard upper bound kept in DB per user (trim oldest when exceeded)


async def get_recent_turns(db, phone_number: str, limit: int = 10) -> List[Dict]:
    """
    Return the last `limit` turns for a user as a list of dicts:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    Returns an empty list when no history exists.
    """
    doc = await db[COLLECTION].find_one({"phone_number": phone_number})
    if not doc or not doc.get("turns"):
        return []

    turns = doc["turns"]
    recent = turns[-limit:] if len(turns) > limit else turns

    flat_history = []
    for turn in recent:
        flat_history.append({"role": "user", "content": turn["user_query"]})
        flat_history.append({"role": "assistant", "content": turn["ai_response"]})

    return flat_history


async def append_turn(db, phone_number: str, user_query: str, ai_response: str) -> None:
    """
    Append one turn to the user's chat history document (upsert).
    Trims the oldest turns if the total exceeds MAX_TURNS.
    """
    now = datetime.now(timezone.utc)

    new_turn = {
        "user_query": user_query,
        "ai_response": ai_response,
        "timestamp": now,
    }

    # Fetch current turn count so we can decide whether to trim.
    doc = await db[COLLECTION].find_one(
        {"phone_number": phone_number}, {"turns": {"$slice": -1}, "turn_count": {"$size": "$turns"}}
    )

    current_count = 0
    if doc:
        # MongoDB $size in projection gives actual count via aggregation;
        # simple find returns the list, so use len() on the fetched slice for the guard.
        all_doc = await db[COLLECTION].find_one({"phone_number": phone_number}, {"turns": 1})
        if all_doc:
            current_count = len(all_doc.get("turns", []))

    if current_count >= MAX_TURNS:
        # Remove the oldest turn, push the new one — stays at MAX_TURNS.
        await db[COLLECTION].update_one(
            {"phone_number": phone_number},
            {
                "$pop": {"turns": -1},  # remove oldest (first element)
            },
        )

    await db[COLLECTION].update_one(
        {"phone_number": phone_number},
        {
            "$push": {"turns": new_turn},
            "$set": {"updated_at": now},
            "$setOnInsert": {"phone_number": phone_number},
        },
        upsert=True,
    )
