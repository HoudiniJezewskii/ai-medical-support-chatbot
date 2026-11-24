"""All prompt text and message-building helpers live here.

"""

SYSTEM_INSTRUCTION = """
You are a concise medical information chatbot.

Rules:
- Answer in no more than 3 short sentences.
- Do not use bullet points or headings.
- Do not say "Okay" or repeat the question.
- Do not diagnose.
- Do not prescribe medication.
- Do not provide extra home remedies.
- If serious symptoms or an emergency are mentioned, recommend immediate medical care.
- Use only information supported by the retrieved sources.
- Do not invent medical information.
- State that the information is general education.
"""

# How many recent messages the bot remembers.
MAX_HISTORY_MESSAGES = 4


def text_message(role, text):
    """Make one chat message in the format MedGemma expects"""
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def format_retrieved_context(retrieved_results):
    """Turn retrieved chunks into one block of text for the prompt"""
    sections = []

    for result in retrieved_results:
        sections.append(
            f"Source: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Information: {result['text']}"
        )

    return "\n\n".join(sections)


def build_augmented_question(question, retrieved_results):
    """Combine the user's question with the retrieved source text."""
    context = format_retrieved_context(retrieved_results)

    return (
        "Use the retrieved medical information below to answer the user's question.\n\n"
        "Retrieved medical information:\n"
        f"{context}\n\n"
        "User question:\n"
        f"{question}"
    )


def build_messages(question, retrieved_results, conversation_history=None):
    """Build the full message list: system rules, recent history, new question."""
    messages = [text_message("system", SYSTEM_INSTRUCTION)]

    if conversation_history:
        messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])

    messages.append(
        text_message("user", build_augmented_question(question, retrieved_results))
    )

    return messages