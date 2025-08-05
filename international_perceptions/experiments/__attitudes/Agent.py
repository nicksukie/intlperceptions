from agentsociety.agent import Agent, AgentParams, AgentContext, BlockOutput
from pydantic import BaseModel, Field
from typing import Optional, Any

class CitizenAgentParams(AgentParams):
    """
    Configuration parameters for CitizenAgent
    """
    exploration_radius: float = Field(default=100.0, description="Maximum exploration radius")
    social_interaction_frequency: int = Field(default=5, description="Social interaction frequency per day")
    energy_consumption_rate: float = Field(default=1.0, description="Energy consumption multiplier")

class CitizenAgentContext(AgentContext):
    """
    Context information for CitizenAgent
    """
    current_goal: str = ""
    mood: str = "neutral"
    daily_schedule: list = Field(default_factory=list)
    visited_locations: list = Field(default_factory=list)

class CitizenBlockOutput(BlockOutput):
    """
    Standardized output format for CitizenAgent blocks
    """
    action_type: str
    success: bool
    energy_cost: float
    message: str = ""

class CitizenAgent(Agent):
    """
    Example CitizenAgent with all class variables defined
    """
    # Class variables directly bound to this agent type
    ParamsType = CitizenAgentParams
    Context = CitizenAgentContext
    BlockOutputType = CitizenBlockOutput
    description: str = "A citizen agent that can explore, interact socially, and manage daily activities"

    async def forward(self):
        """
        Main agent logic implementation
        """
        # Access agent's context
        current_goal = self.context.current_goal