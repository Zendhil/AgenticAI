SUMMARISE_PROMPT = """
Summarise the text in less than 200 words.

Prioritise information that is personally relevant and specific to the user,
rather than generic knowledge or general discussion.

Capture:
- Personal details, preferences, goals, interests, and constraints.
- User-specific experiences, activities, decisions, and opinions.
- Important facts, relationships, and contextual information.
- Commitments, ongoing tasks, plans, and unresolved issues.
- Specific names, dates, figures, technologies, and other details
  that may be useful in future conversations.

Exclude generic explanations, commonly known facts, repetitive content,
and information that does not contribute to understanding the user
or providing more relevant assistance in future interactions.

Preserve the original meaning and important context. Do not invent,
assume, or infer information that is not explicitly stated.

The summary of the given conversation should be concise, factual, and useful as long-term
conversational memory. DO NOT summarise the above instructions.

Conversation:
"""