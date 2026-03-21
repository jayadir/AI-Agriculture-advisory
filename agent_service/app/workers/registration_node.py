from datetime import datetime, timezone
from typing import Optional, Tuple


def parse_registration_sms(message_text: str) -> Optional[Tuple[str, str]]:
    if not isinstance(message_text, str):
        return None

    text = message_text.strip()
    if text.count(",") != 2:
        return None

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        return None

    command, name, pincode = parts
    if command.lower() != "register":
        return None
    if not name or not pincode:
        return None

    return name, pincode


def is_registration_message(message_text: str) -> bool:
    return parse_registration_sms(message_text) is not None


async def register_user_from_sms(db, phone_number: str, message_text: str) -> bool:
    parsed = parse_registration_sms(message_text)
    if not parsed:
        return False

    name, pincode = parsed
    now = datetime.now(timezone.utc)

    await db["users"].update_one(
        {"phone_number": phone_number},
        {
            "$set": {
                "phone_number": phone_number,
                "full_name": name,
                "pincode": pincode,
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
                "crops": [],
            },
        },
        upsert=True,
    )

    return True