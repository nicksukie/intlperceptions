import asyncio
import random
import uuid
from datetime import datetime
from typing import Any, Optional
import csv
import sys
import json
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

from agentsociety.agent import CitizenAgentBase, MemoryAttribute
from agentsociety.cityagent import (
    SocietyAgent,
    default,
)
from agentsociety.configs import (
    AgentsConfig,
    Config,
    EnvConfig,
    ExpConfig,
    LLMConfig,
    MapConfig,
)
from agentsociety.configs.agent import AgentConfig
from agentsociety.configs.exp import (
    AgentFilterConfig,
    WorkflowStepConfig,
    WorkflowType,
)
from agentsociety.environment import EnvironmentConfig
from agentsociety.llm import LLMProviderType
from agentsociety.simulation import AgentSociety
from agentsociety.storage import DatabaseConfig

from agentsociety.agent import AgentToolbox, Block
from agentsociety.memory import Memory
from agentsociety.logger import get_logger
from agentsociety.survey.models import Survey, Page, Question, QuestionType


from attitudeagent import AttitudeAgent
from functools import partial

TIME_SCALE = 2
START_YEAR = 2005
END_YEAR = 2025

def get_param(params, key, default):
    if params is None:
        return default
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)



import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_citizens', type=int, default=100, help='Number of citizen agents')
    parser.add_argument('--memory_file', type=str, default="../gss_twitter_combined_profiles.json", help='Citizen memory file')
    parser.add_argument('--batch_number', type=int, default=1, help='Batches per year for NewsAgent')
    parser.add_argument('--batch_size', type=int, default=50, help='News per batch for NewsAgent')
    parser.add_argument('--news_csv_path', type=str, default="../news_data/news_with_ids.csv", help='Path to news CSV')
    parser.add_argument('--reflection_batch_size', type=int, default=6,
                        help='Number of articles to read before reflecting (for AttitudeAgent)')
    
    return parser.parse_args()

def update_agent_params(agent, params_dict):
    # If agent.agent_params is None, start with dict
    if agent.agent_params is None:
        agent.agent_params = params_dict
    # If it's a dict, update
    elif isinstance(agent.agent_params, dict):
        agent.agent_params.update(params_dict)
    # If it's a Pydantic model, convert to dict, update, and reconstruct
    else:
        # Try to get the model class
        model_cls = type(agent.agent_params)
        new_params = agent.agent_params.dict()
        new_params.update(params_dict)
        agent.agent_params = model_cls(**new_params)

def update_config_from_args(config, args):
    for agent in config.agents.citizens:
        if getattr(agent, "agent_class", None) == AttitudeAgent:
            agent.number = args.num_citizens
            if hasattr(agent, "memory_from_file"):
                agent.memory_from_file = args.memory_file
        if getattr(agent, "agent_class", None) == NewsAgent:
            update_agent_params(agent, {
                "batch_number": args.batch_number,
                "batch_size": args.batch_size,
                "news_csv_path": args.news_csv_path
            })
    return config



def build_config(args):
    # Use args.num_citizens, args.batch_size, etc. here!
    
    # print(f'is the arg being set at all????? {args.reflection_batch_size}')
    
    return Config(
        llm=[
            LLMConfig(
                provider=LLMProviderType.SiliconFlow,
                api_key="[YOUR_API_KEY]",
                model=  "Qwen/Qwen3-14B", #"deepseek-ai/DeepSeek-V3", #,#
                concurrency=10,
                timeout=60,
            )
        ],
        env=EnvConfig(
            db=DatabaseConfig(enabled=True, db_type="sqlite", pg_dsn=None),
            home_dir="../agentsociety_output_data",
        ),
        map=MapConfig(file_path="../agentsociety/beijing_map.pb"),
        agents=AgentsConfig(
            citizens=[
                AgentConfig(
                    agent_class=AttitudeAgent,
                    number=args.num_citizens,
                    memory_from_file=args.memory_file,
                    agent_params={
                        "reflection_batch_size": args.reflection_batch_size,
                        "profile_features_path": args.memory_file,
                    }
                ),
                AgentConfig(
                    agent_class=NewsAgent,
                    number=1,
                    agent_params={
                        "batch_number": args.batch_number,
                        "batch_size": args.batch_size,
                        "news_csv_path": args.news_csv_path
                    }
                ),
            ]
        ),
        exp=ExpConfig(
            name=f"china_opinion_{args.num_citizens}_citizens_{args.batch_number}_batches_{args.batch_size}_batch_size_{args.reflection_batch_size}_reflection_batch_size",
            workflow=[
                WorkflowStepConfig(
                    type=WorkflowType.FUNCTION,
                    func=insert_citizen_information,
                    description="Insert citizen information to news_agent's memory"
                ),
                WorkflowStepConfig(
                    type=WorkflowType.FUNCTION,
                    func=partial(set_reflection_batch_size, value=args.reflection_batch_size),
                    description="Set reflection_batch_size for all AttitudeAgents"
                ),
                WorkflowStepConfig(
                    type=WorkflowType.FUNCTION,
                    func=partial(set_parameters_to_memory, batch_size=args.batch_size, batch_number=args.batch_number, news_csv_path=args.news_csv_path),
                    description="Set parameterss for all News Agents"
                ),
                WorkflowStepConfig(
                    type=WorkflowType.FUNCTION,
                    func=partial(load_user_profiles, file_path=args.memory_file),
                    description="Load user profiles from JSON file"
                ),
                *[
                    step
                    for year in range(START_YEAR, END_YEAR + 1)
                    for step in [
                        WorkflowStepConfig(
                            type=WorkflowType.STEP,
                            steps=1,
                            description=f"Year {year}: Trigger NewsAgent broadcast"
                        ),
                        WorkflowStepConfig(
                            type=WorkflowType.STEP,
                            steps=TIME_SCALE - 1,
                        ),
                        WorkflowStepConfig(
                            type=WorkflowType.SURVEY,
                            survey=china_opinion_survey(),
                            target_agent=AgentFilterConfig(agent_class=(AttitudeAgent,)),
                        ),
                        WorkflowStepConfig(
                            type=WorkflowType.SAVE_CONTEXT,
                            key="china_opinions",
                            save_as=f"opinions_year_{year}",
                            target_agent=AgentFilterConfig(agent_class=(AttitudeAgent,)),
                        ),
                        WorkflowStepConfig(
                            type=WorkflowType.SAVE_CONTEXT,
                            key="selected_news_ids",
                            save_as=f"selected_news_ids_year_{year}",
                            target_agent=AgentFilterConfig(agent_class=(AttitudeAgent,)),
                        ),
                    ]
                ],
            ],
            environment=EnvironmentConfig(
                start_tick=6 * 60 * 60,
            ),
        ),
    )


# NEW: 直接继承CitizenAgentBase，不需要EnvAgentBase（EnvAgentBase是针对BDSC2025设计的）
class NewsAgent(CitizenAgentBase):
    StatusAttributes = [
        MemoryAttribute(name="citizen_ids",type=list,default_or_value=[],description="citizens' ids"),
        MemoryAttribute(name="chat_histories",type=dict,default_or_value={},description="all chat histories"),
        MemoryAttribute(name="batch_size",type=list,default_or_value=[],description="batch size"),
        MemoryAttribute(name="batch_number",type=dict,default_or_value={},description="batch number"),
        MemoryAttribute(name="news_csv_path", type=str, default_or_value="", description="path to news CSV file"),        
        
    ]
    """
    NewsAgent：
    每年第一天通过announcement广播新闻给所有citizens
    """
    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[Any] = None,
        blocks: Optional[list[Block]] = None,
    ):
        super().__init__(id, name, toolbox, memory, agent_params=agent_params, blocks=blocks)
        self.logger = get_logger()
        params = agent_params or {}

        self.logger.info("loading news data from ALL years from CSV file")
        # self.news_by_year = self.load_news(self.news_csv_path)

        self.step = 0
        self.active = True\
            


    async def get_params_from_memory(self):
        """
        Load parameters from memory.
        """
        # Retrieve the parameters dictionary from memory
        params = await self.memory.status.get("agent_params", {})

        # Extract parameters with defaults if not found
        batch_size = params.get("batch_size", 10)
        batch_number = params.get("batch_number", 1)
        news_csv_path = params.get("news_csv_path", "../data/news_data/news_with_ids.csv")

        self.logger.info(
            f"NewsAgent params loaded: batch_size={batch_size}, batch_number={batch_number}, news_csv_path={news_csv_path}"
        )
        return batch_size, batch_number, news_csv_path

    
    async def load_news_for_year(self, file_path: str, year: int) -> list[dict]:
        
        # load parameters from memory

        """Load news for a specific year from CSV."""
        news_this_year = []
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        row_year = int(row["year"])
                        if row_year == year:
                            news_item = {
                                'news_id': row.get('news_id', ''),
                                'title': row.get('title', ''),
                                'full_text': row.get('full_text', ''),
                                'year': row.get('year', ''),
                                # Add other fields as needed
                            }
                            news_this_year.append(news_item)
                    except (KeyError, ValueError):
                        continue
            self.logger.info(f"Loaded {len(news_this_year)} news for year {year}")
            return news_this_year
        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {file_path}")
            return []


    # NEW: 添加广播功能
    async def makeAnnounce(self, content: str, target: Optional[int] = None): # 添加 target 参数
        """
        Makes a city-wide announcement.
        
        - **Description**:
            - Publishes an announcement that reaches all citizens in the city.
            
        - **Args**:
            - `content` (str): The content of the announcement.
            
        - **Returns**:
            - None
        """
        if target is not None:
            await self.send_message_to_agent(target, content)
        else:
            citizen_ids = await self.memory.status.get("citizen_ids")
            for citizen_id in citizen_ids:
                await self.send_message_to_agent(citizen_id, content)
        return {
            "success": True,
            "reason": "You have made an announcement."
        }

    def load_news(self, file_path: str) -> dict[int, list[dict]]:
        """从CSV加载新闻并按年份分组"""
        news_by_year = {}
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        year = int(row["year"][:20]) 
                        news_item = {
                            'news_id': row.get('news_id', ''),  # Add news_id field
                            'title': row.get('title', ''),
                            'full_text': row.get('full_text', ''),
                            'year': row.get('year', ''),
                            # Add any other fields you have in your CSV
                        }
                        news_by_year.setdefault(year, []).append(row)
                    except (KeyError, ValueError):
                        continue
            self.logger.info(f"news_by_year: {news_by_year.keys()}")
            return news_by_year
        except FileNotFoundError:
            self.logger.error(f"找不到新闻文件: {file_path}")
            return {}


    async def forward(self):
        batch_size, batch_number, news_csv_path = await self.get_params_from_memory()
        print(f"NewsAgent params: batch_size={batch_size}, batch_number={batch_number}, news_csv_path={news_csv_path}")
        if not self.active:
            return

        current_year = START_YEAR + self.step // TIME_SCALE
        day_of_year = self.step % TIME_SCALE

        if current_year > END_YEAR:
            if self.active:
                self.logger.info(f"Year {current_year}: 超出新闻年份范围，停止广播")
            self.active = False
            return

        # We will send batch_number batches, distributed throughout the year
        # For example, send batch 0 on day 0, batch 1 on day batch_spacing, etc.
        
        all_news = await self.load_news_for_year(news_csv_path, current_year)
        # if current_year in self.news_by_year:
        #     all_news = self.news_by_year[current_year]
        citizen_ids = await self.memory.status.get("citizen_ids")
        if not citizen_ids:
            self.logger.warning("没有有效的 citizen_ids，无法继续广播新闻")
            return

        # Calculate spacing between batches (avoid division by zero)
        batch_spacing = max(1, TIME_SCALE // batch_number)
        get_logger().info(f"Year {current_year}: Broadcasting news in {batch_number} batches, each with {batch_size} articles, spaced every {batch_spacing} days.")

        # Send batch only on the scheduled days
        if day_of_year % batch_spacing == 0 and (day_of_year // batch_spacing) < batch_number:
            batch_idx = day_of_year // batch_spacing
            self.logger.info(f"Year {current_year}: Broadcasting news batch {batch_idx+1}/{batch_number}")

            # # Sample a batch (each batch is random, not necessarily unique)
            # batch_news = random.sample(all_news, min(batch_size, len(all_news)))

            for citizen_id in citizen_ids:
                # Each agent gets a random subset from this batch
                exposed_news = random.sample(all_news, min(batch_size, len(all_news)))
                # news_payload = {
                #     "news_id": news['news_id'],
                #     "title": news['title'],
                #     "full_text": news['full_text']
                # }
                content = f"news:{json.dumps(exposed_news)}"
                result = await self.makeAnnounce(content, target=citizen_id)
                if result["success"]:
                    self.logger.info(f"广播成功: targeted agent: {citizen_id} with {len(exposed_news)} articles ")
                else:
                    self.logger.warning(f"广播失败: {result['reason']}")
        else:
            if day_of_year == 0:
                self.logger.debug(f"没有 {current_year} 年的新闻数据")

        self.step += 1




def china_opinion_survey(initial: bool = False) -> Survey:
    """
    问卷：收集agents阅读新闻前后对中国态度的变化
    """
    survey_id = uuid.uuid4()
    title = "attitude on China After Reading News"
    description = "This survey collects your opinion on China after reading the assigned news articles."

    questions = [
        Question(
            name="china_opinion",
            title = (
                "Now that you've read the above news articles, please answer the following question:\n\n"
                "On a scale from 1-4 where:\n"
                "    1 - Very unfavorable\n\n"
                "    2 - Somewhat unfavorable\n"
                "    3 - Somewhat favorable\n"
                "    4 - Very favorable\n"
                "How would you rate your current opinion of China? Please respond with a number "
                "(1, 2, 3, 4)\n\n"
                "**High means positive!**"
        ),
            type=QuestionType.RATING,
            min_rating=1,
            max_rating=4,
            required=True,
        ),
    ]
    page = Page(name="china_attitude", elements=questions)
    return Survey(
        id=survey_id,
        title=title,
        description=description,
        pages=[page],
        created_at=datetime.now()
    )

"""
将news和attitude写入 stream memory
async def on_announcement(self, content: str):
    await self.memory.stream.add_event(
        tag="NEWS",
        day=self.environment.get_tick(),
        description=f"Received news: {content[:50]}..."  # 只写 headline 前50字
    )
    attitude = self.memory.attributes.get('attitude', 'unknown')
    await self.memory.stream.add_event(
        tag="ATTITUDE",
        day=self.environment.get_tick(),
        description=f"Current attitude: {attitude}"
    )
"""


async def load_user_profiles(simulation: AgentSociety, file_path: str):
    """
    Load user profiles from a JSON file and update the memory of AttitudeAgents,
    splitting the profile into demographics, domestic views, media habits, interests, and personality summary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    demographics_fields = [
        "sex", "race", "marital", "degree", "educ", "region", "income", "class", "age",
        "partyid", "polviews", "leftrght"
    ]
    media_fields = [
        "polnews", "polnewsfrom", "smnews", "tvnews1", "papernews", "radionews", "webnews"
    ]
    domestic_views_fields = [
        "lkelyvot", "polintrst", "cantrust", "befair", "corruptn", "oppsegov", "rghtsmin", "fulldem",
        "eqwlth", "tax", "gunlaw", "cappun", "nataid", "natspac", "natenvir", "natheal", "imports",
        "immcrime", "immjobs", "immameco", "class", "union1", "satfin", "grass", "conlegis", "conjudge",
        "conarmy", "wrldgovt"
    ]

    attitude_agent_ids = await simulation.filter(types=(AttitudeAgent,))
    print(f'attitude_agent_ids: {attitude_agent_ids}')
    for idx, agent_id in enumerate(attitude_agent_ids):
        print('loading profile for agent:', agent_id)
        if idx < len(profiles):
            profile = profiles[idx]
            demographics = {k: profile.get(k) for k in demographics_fields if k in profile}
            media_habits = {k: profile.get(k) for k in media_fields if k in profile}
            domestic_views = {k: profile.get(k) for k in domestic_views_fields if k in profile}
            interests = profile.get("interests", [])
            personality_summary = profile.get("personality_summary", "")

            # Update the agent's memory with the extracted data
            await simulation.update([agent_id], "demographics", demographics)
            await simulation.update([agent_id], "media_habits", media_habits)
            await simulation.update([agent_id], "domestic_views", domestic_views)
            await simulation.update([agent_id], "interests", interests)
            await simulation.update([agent_id], "personality_summary", personality_summary)
            # print(f"the user's profile is loaded in the config file: {demographics}, {media_habits}, {domestic_views}, {interests}, {personality_summary}")
        else:
            print(f"Profile for agent {agent_id} not found in the file (idx {idx} out of range).")


# New: Insert citizen ids so that you can broadcast news to them
async def insert_citizen_information(simulation: AgentSociety):
    """
    Insert the citizen information into the news_agent's memory.
    """
    citizen_ids = await simulation.filter(types=(AttitudeAgent,))
    print(f"citizen_ids: {citizen_ids}")
    news_agent = (await simulation.filter(types=(NewsAgent,)))[0]
    await simulation.update([news_agent], "citizen_ids", citizen_ids)
    

async def set_reflection_batch_size(simulation: AgentSociety, value: int):
    """
    Set the reflection_batch_size attribute on all AttitudeAgents.
    """
    agent_ids = await simulation.filter(types=(AttitudeAgent,))
    await simulation.update(agent_ids, "reflection_batch_size", value)
    print(f"Set reflection_batch_size to {value} for all AttitudeAgents.")\
        
        
async def set_parameters_to_memory(simulation: AgentSociety, batch_size: int, batch_number: int, news_csv_path: str):
    """
    Set the batch_size, batch_number, and news_csv_path parameters in the memory of NewsAgent.
    """
    agent_ids = await simulation.filter(types=(NewsAgent,))
    params = {
        "batch_size": batch_size,
        "batch_number": batch_number,
        "news_csv_path": news_csv_path
    }
    await simulation.update(agent_ids, "agent_params", params)
    print(f"Set parameters {params} for all NewsAgents.")

async def main():
    args = parse_args()
    config = build_config(args)
    agentsociety = AgentSociety.create(config)
    try:
        await agentsociety.init()
        await agentsociety.run()
    finally:
        await agentsociety.close()

if __name__ == "__main__":
    asyncio.run(main())