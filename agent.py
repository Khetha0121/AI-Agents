import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    MODEL_NAME = os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.0-flash")
except ImportError:
    print("Warning: python-dotenv not installed. Ensure API key is set")
    MODEL_NAME = "gemini-2.0-flash"

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

from marketing_agents.instructions import (
    MARKET_RESEARCH_INSTRUCTION,
    MESSAGING_STRATEGIST_INSTRUCTION,
    AD_COPY_WRITER_INSTRUCTION,
    VISUAL_SUGGESTER_INSTRUCTION,
    FORMATTER_INSTRUCTION,
    CAMPAIGN_ORCHESTRATOR_INSTRUCTION
)

# TOOL REGISTRY

# Multiple tool integrations
class ToolRegistry:
    @staticmethod
    def get_research_tools():
        return [
            google_search,
        ]

# --- Base Agent class ---
class MarketingLlmAgent(LlmAgent):
    def __init__(self, name, instruction, output_key, tools=None):
        super().__init__(
            name=name,
            model=MODEL_NAME,
            instruction=instruction,
            output_key=output_key,
            temperature = 0.3,
            tools=tools or []
        )

# --- Sub Agents ---
market_research_agent = MarketingLlmAgent(
    name="MarketResearcher",
    instruction=MARKET_RESEARCH_INSTRUCTION,
    output_key="market_research_summary",
    tools=ToolRegistry.get_research_tools()
)

messaging_strategist_agent = MarketingLlmAgent(
    name="MessagingStrategist",
    instruction=MESSAGING_STRATEGIST_INSTRUCTION,
    output_key="key_messaging"
)

ad_copy_writer_agent = MarketingLlmAgent(
    name="AdCopyWriter",
    instruction=AD_COPY_WRITER_INSTRUCTION,
    output_key="ad_copy_variations"
)

visual_suggester_agent = MarketingLlmAgent(
    name="VisualSuggester",
    instruction=VISUAL_SUGGESTER_INSTRUCTION,
    output_key="visual_concepts"
)

formatter_agent = MarketingLlmAgent(
    name="CampaignBriefFormatter",
    instruction=FORMATTER_INSTRUCTION,
    output_key="final_campaign_brief"
)

campaign_orchestrator = SequentialAgent(
    name="MarketingCampaignAssistant",
    description=CAMPAIGN_ORCHESTRATOR_INSTRUCTION,
    sub_agents=[
        market_research_agent,
        messaging_strategist_agent,
        ad_copy_writer_agent,
        visual_suggester_agent,
        formatter_agent,
    ]
)

root_agent = campaign_orchestrator
