"""Shared task definition: prompt, label rules, and output parsing.

Intent labels come from Banking77. `urgency` and `needs_human` are derived
deterministically from the intent (see rules below), so the eval is fully
objective. They are rule-based labels, not human annotations.
"""
import json
import re

SYSTEM_PROMPT = (
    "You are a banking support triage assistant. Read the customer message and "
    "reply with ONLY a JSON object with keys: "
    '"intent" (snake_case label), "urgency" (low|medium|high), '
    '"needs_human" (true|false).'
)

HIGH_URGENCY = {
    "lost_or_stolen_card", "lost_or_stolen_phone", "compromised_card",
    "card_swallowed", "card_payment_not_recognised",
    "cash_withdrawal_not_recognised", "direct_debit_payment_not_recognised",
    "transaction_charged_twice",
}
MEDIUM_URGENCY = {
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "failed_transfer", "pending_card_payment", "pending_cash_withdrawal",
    "pending_transfer", "pending_top_up", "top_up_failed", "top_up_reverted",
    "transfer_not_received_by_recipient", "wrong_amount_of_cash_received",
    "request_refund", "refund_not_showing_up", "card_not_working",
    "virtual_card_not_working", "contactless_not_working", "pin_blocked",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "extra_charge_on_statement", "card_payment_fee_charged",
    "unable_to_verify_identity",
}
ALWAYS_HUMAN = {
    "terminate_account", "request_refund", "refund_not_showing_up",
    "transfer_not_received_by_recipient", "wrong_amount_of_cash_received",
}


def normalize_intent(name: str) -> str:
    return name.strip().lower()


def urgency_for(intent: str) -> str:
    if intent in HIGH_URGENCY:
        return "high"
    if intent in MEDIUM_URGENCY:
        return "medium"
    return "low"


def needs_human_for(intent: str) -> bool:
    return urgency_for(intent) == "high" or intent in ALWAYS_HUMAN


def make_target(intent: str) -> dict:
    intent = normalize_intent(intent)
    return {
        "intent": intent,
        "urgency": urgency_for(intent),
        "needs_human": needs_human_for(intent),
    }


def parse_output(text: str):
    """Return the parsed dict if `text` is valid JSON with the right keys, else None."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj) != {"intent", "urgency", "needs_human"}:
        return None
    if obj["urgency"] not in {"low", "medium", "high"}:
        return None
    if not isinstance(obj["needs_human"], bool):
        return None
    return obj


def zero_shot_system(intents) -> str:
    """System prompt for the no-fine-tuning baseline: the model can't know the
    77 labels unless we list them."""
    return SYSTEM_PROMPT + " Valid intent labels: " + ", ".join(sorted(intents)) + "."
