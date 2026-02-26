"""
Agente LangChain — Conecta ao QWEN 4B via LM Studio para gerar análises textuais.

Usa a API OpenAI-compatible do LM Studio para enviar os dados de tracking
e receber textos analíticos para o relatório PDF.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def get_llm(lm_studio_url: str = "http://26.198.160.131:1234/v1") -> ChatOpenAI:
    """
    Cria uma instância do LLM conectado ao LM Studio.
    
    Args:
        lm_studio_url: URL base da API do LM Studio.
    """
    return ChatOpenAI(
        base_url=lm_studio_url,
        api_key="lm-studio",  # LM Studio aceita qualquer chave
        model="qwen2.5-4b",
        temperature=0.7,
        max_tokens=2048,
    )


ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a professional sports marketing analyst specializing in Formula 1 "
        "advertising and brand visibility analysis. You write reports in Portuguese (Brazil). "
        "Be concise, data-driven, and provide actionable insights. "
        "Format your response with clear sections using markdown."
    )),
    ("human", (
        "Analise os seguintes dados de detecção de marcas em um vídeo de Fórmula 1 "
        "e gere um relatório analítico com:\n\n"
        "1. **Resumo Executivo** — visão geral do vídeo e das detecções\n"
        "2. **Análise da Marca Dominante** — por que essa marca se destaca e possíveis razões\n"
        "3. **Tendências e Observações** — padrões notáveis nos dados\n"
        "4. **Recomendações** — sugestões baseadas nos dados\n\n"
        "Dados de análise:\n\n{metrics_text}"
    )),
])


def generate_analysis(metrics_text: str, lm_studio_url: str = "http://26.198.160.131:1234/v1") -> str:
    """
    Gera uma análise textual usando o agente LLM.
    
    Args:
        metrics_text: Texto formatado com as métricas do BrandTracker.
        lm_studio_url: URL base da API do LM Studio.
        
    Returns:
        Texto da análise gerada pelo agente.
    """
    llm = get_llm(lm_studio_url)
    chain = ANALYSIS_PROMPT | llm
    response = chain.invoke({"metrics_text": metrics_text})
    return response.content


if __name__ == "__main__":
    # Teste rápido de conexão
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test LM Studio connection")
    parser.add_argument("--url", type=str, default="http://26.198.160.131:1234/v1")
    args = parser.parse_args()

    if args.test:
        print("Testing connection to LM Studio...", flush=True)
        try:
            llm = get_llm(args.url)
            response = llm.invoke("Diga 'conexão OK' se estiver funcionando.")
            print(f"✅ Connection successful! Response: {response.content}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    else:
        # Test with mock data
        mock_metrics = """=== Video Analysis Results ===
Total Frames: 9000
FPS: 30.0
Duration: 300.0s
Total Brands Detected: 5
Most Frequent Brand: Heineken (visible in 4500 frames)

=== Per-Brand Breakdown ===

[Heineken]
  Frames Visible: 4500
  Time Visible: 150.0s (50.0%)
  Total Detections: 8200
  Avg Confidence: 0.8523

[Rolex]
  Frames Visible: 3200
  Time Visible: 106.67s (35.56%)
  Total Detections: 5100
  Avg Confidence: 0.7891
"""
        print("Generating analysis with mock data...", flush=True)
        result = generate_analysis(mock_metrics, args.url)
        print(result)
