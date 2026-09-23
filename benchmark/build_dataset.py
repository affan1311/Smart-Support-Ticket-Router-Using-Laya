"""Build the labeled benchmark set: data/benchmark_labeled.csv.

Three parts, chosen after checking label quality in each source:

1. "bitext": 240 tickets (60 each: account, billing, shipping, general) from
   bitext/Bitext-customer-support-llm-chatbot-training-dataset (CDLA-Sharing-1.0).
   Clean e-commerce intent labels, mapped to departments below. Short, chat-style
   messages, some with slang, typos or swearing. No technical intents and no urgency labels.

2. "tobi": 60 technical tickets from Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0).
   That dataset's queue labels are noisy (its "Billing" and "Returns" queues are mostly
   IT tickets), so only technical-queue tickets whose tags also say technical are kept.
   Its priority field gives urgency labels.

3. "hard": data/benchmark_hard.csv, ~45 hand-written tickets with all 5 labels
   (threats, fraud, abuse, sarcasm, account takeover, very short and very long messages).

Policy/human/standard-reply labels exist only for the hard set. Public tickets are
assumed to have no policy violation (swearing isn't one), which measures false alarms.

Usage (from the repo root):  .venv\\Scripts\\python benchmark\\build_dataset.py
"""
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
PER_DEPARTMENT = 60
SEED = 42

# Bitext intent -> department. Order changes (cancel/change/place order) are left out:
# they could reasonably go to shipping, billing or general, so their labels would be arbitrary.
INTENT_TO_DEPARTMENT = {
    "create_account": "account", "delete_account": "account", "edit_account": "account",
    "recover_password": "account", "registration_problems": "account", "switch_account": "account",
    "check_cancellation_fee": "billing", "check_invoice": "billing", "get_invoice": "billing",
    "check_payment_methods": "billing", "payment_issue": "billing", "check_refund_policy": "billing",
    "get_refund": "billing", "track_refund": "billing",
    "delivery_options": "shipping", "delivery_period": "shipping", "track_order": "shipping",
    "change_shipping_address": "shipping", "set_up_shipping_address": "shipping",
    "contact_customer_service": "general", "contact_human_agent": "general", "complaint": "general",
    "review": "general", "newsletter_subscription": "general",
}

# Bitext messages contain template slots like {{Order Number}}; fill them with realistic values.
PLACEHOLDERS = {
    "Order Number": "#48213", "Account Type": "premium", "Person Name": "Alex", "Account Category": "business",
    "Refund Amount": "$49", "Currency Symbol": "$", "Delivery City": "Leeds", "Delivery Country": "Canada",
    "Invoice Number": "INV-2291",
}

TOBI_TECH_QUEUES = {"Technical Support", "IT Support", "Service Outages and Maintenance"}
TOBI_TECH_TAGS = {"Bug", "Outage", "Crash", "Disruption", "Network", "Hardware", "Software"}
TOBI_NOT_TECH_TAGS = {"Billing", "Payment", "Refund", "Sales", "Marketing", "Digital Marketing", "Lead",
                      "Strategy", "Digital Strategy", "Campaign", "Analytics", "Social Media"}
PRIORITY_TO_URGENCY = {"low": 0, "medium": 1, "high": 2}


def fill_placeholders(text: str) -> str:
    return re.sub(r"\{\{(.*?)\}\}", lambda m: PLACEHOLDERS.get(m.group(1), m.group(1).lower()), text)


def load_bitext() -> pd.DataFrame:
    path = hf_hub_download("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                           "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
                           repo_type="dataset")
    df = pd.read_csv(path)
    df = df[df["intent"].isin(INTENT_TO_DEPARTMENT)].drop_duplicates(subset=["instruction"]).copy()
    df["department"] = df["intent"].map(INTENT_TO_DEPARTMENT)

    # Same count per department, spread evenly over that department's intents.
    parts = []
    for _, group in df.groupby("department"):
        intents = sorted(group["intent"].unique())
        per_intent = -(-PER_DEPARTMENT // len(intents))  # ceiling division
        sample = group.groupby("intent", group_keys=False).sample(n=per_intent, random_state=SEED)
        parts.append(sample.sample(n=PER_DEPARTMENT, random_state=SEED))
    sample = pd.concat(parts)
    return pd.DataFrame({
        "set": "bitext",
        "subject": "(no subject)",  # chat messages have no subject line
        "body": sample["instruction"].map(fill_placeholders),
        "department": sample["department"],
        "urgency": "",  # unknown
        "policy_violation": "no",
        "human_needed": "",
        "standard_reply": "",
        "source_label": sample["intent"],
        "note": "",
    })


def load_tobi_technical() -> pd.DataFrame:
    path = hf_hub_download("Tobi-Bueck/customer-support-tickets",
                           "aa_dataset-tickets-multi-lang-5-2-50-version.csv", repo_type="dataset")
    df = pd.read_csv(path)
    df = df[(df["language"] == "en") & df["queue"].isin(TOBI_TECH_QUEUES)].dropna(subset=["body"])
    df = df.drop_duplicates(subset=["body"]).copy()
    tags = df[[f"tag_{i}" for i in range(1, 9)]].apply(
        lambda row: {str(t).strip() for t in row if pd.notna(t)}, axis=1)
    df = df[tags.apply(lambda s: bool(s & TOBI_TECH_TAGS) and not (s & TOBI_NOT_TECH_TAGS))]
    sample = df.sample(n=PER_DEPARTMENT, random_state=SEED)
    return pd.DataFrame({
        "set": "tobi",
        "subject": sample["subject"].fillna("(no subject)"),
        "body": sample["body"],
        "department": "technical",
        "urgency": sample["priority"].map(PRIORITY_TO_URGENCY).astype(int).astype(str),
        "policy_violation": "no",
        "human_needed": "",
        "standard_reply": "",
        "source_label": sample["queue"] + " / " + sample["priority"],
        "note": "",
    })


def load_hard() -> pd.DataFrame:
    hard = pd.read_csv(DATA / "benchmark_hard.csv", dtype=str, keep_default_na=False)
    hard.insert(0, "set", "hard")
    hard["source_label"] = ""
    return hard


def main() -> None:
    labeled = pd.concat([load_bitext(), load_tobi_technical(), load_hard()], ignore_index=True)
    labeled.insert(0, "id", range(1, len(labeled) + 1))
    out = DATA / "benchmark_labeled.csv"
    labeled.to_csv(out, index=False)

    print(f"Wrote {len(labeled)} tickets to {out}")
    print(labeled.groupby(["set", "department"]).size().unstack(fill_value=0).to_string())
    print("\nUrgency labels:", labeled.loc[labeled["urgency"] != "", "urgency"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
