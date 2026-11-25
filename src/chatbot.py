"""The full question -> answer pipeline

"""

from src.models import generate_text
from src.prompts import build_messages, text_message


def answer_question(
    question,
    model,
    processor,
    retriever,
    conversation_history=None,
    number_of_results=3,
    min_score=0.0,
    max_new_tokens=70,
):
    """Answer one question using retrieved sources.

    Returns a dictionary:
      - "answer":  the model's text
      - "sources": the retrieved chunks (title, url, tier, score, text)
    """
    retrieved = retriever.retrieve(
        question,
        number_of_results=number_of_results,
        min_score=min_score,
    )

    messages = build_messages(question, retrieved, conversation_history)

    answer = generate_text(
        model,
        processor,
        messages,
        max_new_tokens=max_new_tokens,
    )

    return {"answer": answer, "sources": retrieved}


def add_to_history(conversation_history, question, answer):
    """Save the users ORIGINAL question and the bots answer.

    The long retrieved text is not saved, which keeps the memory short.
    """
    conversation_history.append(text_message("user", question))
    conversation_history.append(text_message("assistant", answer))


def format_sources(sources):
    """Make a short, readable list of sources to show the user."""
    seen = set()
    lines = []

    for source in sources:
        if source["url"] in seen:
            continue
        seen.add(source["url"])
        lines.append(f"- {source['title']}: {source['url']}")

    return "\n".join(lines)