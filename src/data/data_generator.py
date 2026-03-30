import json

intents = [
    "Read file",
    "Open browser",
    "Download data",
    "Send email",
    "Access system",
    "Play music",
    "Check logs",
    "Backup data"
]

safe_actions = [
    "read content",
    "open homepage",
    "download file",
    "send email",
    "view system info",
    "play song",
    "view logs",
    "backup files"
]

malicious_suffixes = [
    "and send to external server",
    "and delete logs",
    "and execute hidden script",
    "and escalate privileges",
    "and upload data online",
    "and steal credentials",
    "and modify system settings"
]

data = []

# SAFE CASES
for i in range(len(intents)):
    data.append({
        "intent": intents[i],
        "action": intents[i] + " " + safe_actions[i],
        "label": 1,
        "category": "safe"
    })

# MALICIOUS CASES
for intent in intents:
    for suffix in malicious_suffixes:
        data.append({
            "intent": intent,
            "action": intent + " " + suffix,
            "label": 0,
            "category": "attack"
        })

# SAVE
with open("src/data/generated_cases.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(data)} test cases!")