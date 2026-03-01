"""
ai_agent.py — LangChain agent that generates AI-powered analysis of brand metrics.

Connects to a local LLM (Qwen 4B) running on LM Studio to produce a written
analysis of brand visibility data. The analysis is included in the PDF report.

Requirements:
    - LM Studio running locally with a model loaded (e.g., Qwen 4B)
    - langchain and langchain-openai packages installed
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# System prompt that defines the AI agent's role and output format
SYSTEM_PROMPT = """You are an expert F1 marketing analyst. You will receive brand visibility
metrics from a Formula 1 video analysis and generate a professional report.

Your analysis should cover:
1. **Overview**: A brief summary of which brands dominate screen time.
2. **Key Findings**: The most visible brands and their significance.
3. **Comparison**: How brands compare in terms of visibility and confidence scores.
4. **Insights**: What the data suggests about sponsorship placement strategy.

Write in a professional, data-driven style. Use specific numbers from the data.
Keep the analysis concise but insightful (300-500 words).
Format your response in Markdown with headers (##) and bullet points."""


def generate_analysis(metrics_text: str,
                       lm_studio_url: str = "http://26.198.160.131:1234/v1") -> str:
    """
    Generate an AI-written analysis of the brand visibility metrics.

    Sends the metrics data to a local LLM via LM Studio and returns
    a formatted analysis suitable for the PDF report.

    Args:
        metrics_text: Human-readable metrics summary from BrandTracker.
        lm_studio_url: URL of the LM Studio OpenAI-compatible API.

    Returns:
        AI-generated analysis text in Markdown format.

    Raises:
        Exception: If LM Studio is unreachable or the model fails to respond.
    """
    # Connect to the local LLM via LM Studio's OpenAI-compatible API
    llm = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="lm-studio",         # LM Studio doesn't require a real key
        model="qwen/qwen3-vl-4b",
        temperature=0.7,
        max_tokens=2048,
    )

    # Build the prompt with system instructions + metrics data
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Here are the brand visibility metrics from the F1 video analysis:\n\n{metrics}\n\nPlease generate a professional analysis report."),
    ])

    # Create and invoke the chain
    chain = prompt | llm
    response = chain.invoke({"metrics": metrics_text})

    return response.content


if __name__ == '__main__':
    # Quick test: run with sample data
    sample = """
    [MSC cruises]  Time: 190.92s (39.15%)  Detections: 37471  Confidence: 0.5622
    [Qatar Airways] Time: 36.16s (7.42%)   Detections: 3491   Confidence: 0.6762
    [Pirelli]       Time: 33.24s (6.82%)   Detections: 2588   Confidence: 0.6120
    """
    print("Testing AI agent...")
    result = generate_analysis(sample)
    print(result)
