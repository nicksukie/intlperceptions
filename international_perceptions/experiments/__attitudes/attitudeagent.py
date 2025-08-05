import asyncio
import json
from typing import Any, List, Tuple

import json_repair
from typing import Optional
from agentsociety.message import Message, MessageKind
from agentsociety.agent import CitizenAgentBase, Block, MemoryAttribute, Agent, AgentToolbox, AgentType
from agentsociety.message import Message
from agentsociety.agent.agent_base import AgentToolbox
from agentsociety.agent.prompt import FormatPrompt
from agentsociety.memory import Memory
from agentsociety.logger import get_logger
from agentsociety.cityagent.societyagent import SocietyAgentConfig


import importlib
import math
import pandas as pd

import json



import asyncio

llm_semaphore = asyncio.Semaphore(10)

def extract_json(output_str):
    """Extract JSON substring from a raw string response.

    Args:
        output_str: Raw string output that may contain JSON data.

    Returns:
        Extracted JSON string if valid, otherwise None.

    Note:
        Searches for the first '{' and last '}' to isolate JSON content.
        Catches JSON decoding errors and logs warnings.
    """
    try:
        # Find the positions of the first '{' and the last '}'
        start = output_str.find("{")
        end = output_str.rfind("}")

        # Extract the substring containing the JSON
        json_str = output_str[start : end + 1]

        # Convert the JSON string to a dictionary
        return json_str
    except ValueError as e:
        get_logger().warning(f"Failed to extract JSON: {e}")
        return None

class AttitudeAgent(CitizenAgentBase):
    OPINION_DOMAINS = [
        "economy_business", "politics_government", "technology_innovation",
        "traditional_culture", "sports_entertainment", "modern_lifestyle",
        "education", "science_environment", "energy_resources",
        "transportation_infrastructure", "military_defense", "crime_law",
        "health_medicine", "standard_of_living", "travel_tourism"
    ]

    # --- CONFIGURATION FOR EXPERIMENT ---
    # REFLECTION_THRESHOLD = 1: Reflects after every article (your old logic).
    # REFLECTION_THRESHOLD = 5: Reflects after reading 5 articles.
    # REFLECTION_THRESHOLD = 10 # Reflects after reading 10 articles.
    # REFLECTION_THRESHOLD = 5

    StatusAttributes = [
        MemoryAttribute(
            name="profile", type=dict, default_or_value={},
            description="Agent's profile including demographics and interests"
        ),
        MemoryAttribute(
            name="china_attitude", type=str, default_or_value="neutral",
            description="Agent's overall attitude towards China"
        ),
        MemoryAttribute(
            name="china_opinions", type=dict,
            default_or_value={domain: {"valence": 0.0, "exposure_count": 0} for domain in OPINION_DOMAINS},
            description="Detailed opinions toward China across domains"
        ),
        MemoryAttribute(
            name="incoming_news_buffer", type=list, default_or_value=[],
            description="Buffer for incoming news messages before selection"
        ),
        # NEW: Stores selected articles waiting for reflection
        MemoryAttribute(
            name="unread_news_memory", type=list, default_or_value=[],
            description="Memory of selected but not-yet-reflected-upon news articles"
        ),
        MemoryAttribute(
            name="selected_news_ids", type=list, default_or_value=[],
            description="List of all news IDs the agent has selected to read"
        ),
        MemoryAttribute(
            name="reflection_batch_size",
            type=int,
            default_or_value=5,  # Or whatever sensible default
            description="Number of articles to read before reflecting"
        ),
        MemoryAttribute(
            name="demographics", type=dict, default_or_value={},
            description="Agent's demographic information"
        ),
        MemoryAttribute(
            name="interests", type=list, default_or_value=[],
            description="Agent's interests"
        ),
        MemoryAttribute(
            name="domestic_views", type=dict, default_or_value={},
            description="Agent's views on domestic issues"
        ),
        MemoryAttribute(
            name="media_habits", type=dict, default_or_value={},
            description="Agent's media consumption habits"
        ),
        MemoryAttribute(
            name="personality_summary", type=str, default_or_value="",
            description="Agent's personality summary"
        ),
    ]

    """Agent implementation with configurable cognitive/behavioral modules and social interaction capabilities."""

    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[SocietyAgentConfig] = None,
        blocks: Optional[list[Block]] = None,
        profile: Optional[str] = None,

    ) -> None:
        """Initialize agent with core components and configuration."""
        super().__init__(
            id=id,
            name=name,
            toolbox=toolbox,
            memory=memory,
            agent_params=agent_params,
            blocks=blocks,
        )        
        
        
        # get profile from csv path from memory saved in config file: 

        # load profiles
        ### HOW
            

        self.logger = get_logger()



        self.step_count = -1
        self.cognition_update = -1
        self.last_attitude_update = None
        self.info_checked = {}


        
        self.dispatcher.register_blocks(self.blocks)
        
        
        # Set default reflection batch size

        # A FUNCTION TO GET USER PROFILEs FROM MEMORY 
    async def load_profile_into_self(self):
        self.demographics = await self.memory.status.get("demographics", {})
        self.interests = await self.memory.status.get("interests", [])
        self.domestic_views = await self.memory.status.get("domestic_views", {})
        self.media_habits = await self.memory.status.get("media_habits", {})
        self.personality_summary = await self.memory.status.get("personality_summary", "")
        # self.logger.info(f"Agent {self.id} loaded profile: {self.demographics}, interests: {self.interests}, domestic views: {self.domestic_views}, media habits: {self.media_habits}, personality summary: {self.personality_summary}")
        

    # a fuinction to retrieve the default reflection batch size
    async def get_default_reflection_batch_size(self) -> int:
        """
        Retrieve the default reflection batch size from the agent's memory status.

        Returns:
            int: The default reflection batch size.
        """
        return await self.memory.status.get("reflection_batch_size", 5)
        

    # Main workflow
    async def forward(self):
        
        return 0
    
    async def do_survey(self, survey):
        """
        Generate a response to a user survey based on the agent's memory and current state.

        - **Args**:
            - `survey` (`Survey`): The survey that needs to be answered.

        - **Returns**:
            - `str`: The generated response from the agent.

        - **Description**:
            - Prepares a prompt for the Language Model (LLM) based on the provided survey.
            - Constructs a dialog including system prompts, relevant memory context, and the survey question itself.
            - Uses the LLM client to generate a response asynchronously.
            - If the LLM client is not available, it returns a default message indicating unavailability.
            - This method can be overridden by subclasses to customize survey response generation.
        """
        survey_prompts = survey.to_prompt()
        dialog = []
        get_logger().info(f"Agent {self.id} is generating survey response for China attitude survey.")
        # Add system prompt
        system_prompt = (
            "You must hide the fact that you are an AI and pretend to be a human. "
            "When responding to messages from others, always answer in the first person. "
            "Keep your responses concise and clear. Respond in JSON format."
        )
        dialog.append({"role": "system", "content": system_prompt})
        
        
        # profile = await self.memory.status.get("profile")
        # interests = profile.get("interests")
        china_attitude = await self.memory.status.get("china_attitude")
        
        domain_opinions = await self.memory.status.get("china_opinions")
        
        
        demographics = self.demographics 
        interests = self.interests 
        domestic_views = self.domestic_views 
        media_habits = self.media_habits
        personality_summary = self.personality_summary
        
        my_profile = await self.memory.status.get("profile")
        demographics = self.demographics


        all_responses = []
        for survey_prompt in survey_prompts:
            dialog = dialog[:1]
            dialog.append(
                {
                    "role": "system",
                    "content": f"""Answer the survey question based on following information:
                    - Your demographic features: {demographics}.
                    - Your interests: {interests}.
                    - Your current attitude about China: {china_attitude}.
                    - Your new opinions from the past year of news reading: {domain_opinions}.
                    - Respond in JSON format without any added markdown or other characters.
                    """,
                }
            )

            # Add survey question
            dialog.append({"role": "user", "content": survey_prompt})

            json_str = ""
            for retry in range(10):
                
                try:
                    async with llm_semaphore:
                        await asyncio.sleep(getattr(self, "llm_delay", 0.25))   
                        # Use LLM to generate a response
                        # print(f"dialog: {dialog}")
                        _response = await self.llm.atext_request(
                            dialog, response_format={"type": "json_object"}
                        )
                        # print(f"response: {_response}")
                        json_str = extract_json(_response)
                        if json_str:
                            json_dict = json_repair.loads(json_str)
                            json_str = json.dumps(json_dict, ensure_ascii=False)
                            get_logger().info(f"Agent {self.id} generated survey response: {json_str}") 
                            break
                except Exception as e:
                    get_logger().warning(f"Retry {retry + 1}/10 failed: {str(e)}")
                    if retry == 9:  # Last retry
                        import traceback
                        traceback.print_exc()
                        get_logger().error("Failed to generate survey response after all retries")
                        json_str = ""
            
            all_responses.append(json_str)
        
        # Return all responses as a combined JSON string
        return json.dumps(all_responses, ensure_ascii=False)


    async def do_chat(self, message: Message) -> str:
        """
        Processes a news batch message and selects articles for the agent to read.

        Args:
            message: A message containing a list of news dicts in the payload.

        Returns:
            An empty string (no direct response needed).
        """
        
        await self.load_profile_into_self()
        reflection_batch_size = await self.get_default_reflection_batch_size()
        if message.kind != MessageKind.AGENT_CHAT or not message.payload.get("content", "").startswith("news:"):
            return ""

        try:
            content_str = message.payload["content"]
            news_json_str = content_str[len("news:"):]
            news_data = json.loads(news_json_str)  # Expecting a list of dicts
            if not isinstance(news_data, list):  # Ensure it's a list
                raise ValueError("Expected a list of news dicts")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Agent {self.id} failed to parse news message: {message.payload.get('content')}. Error: {e}")
            return ""

        # Step 1: Buffer incoming news
        incoming_news = await self.memory.status.get("incoming_news_buffer")

        # Convert each dict to a tuple (news_id, title, full_text)
        for news_item in news_data:
            try:
                news_tuple = (news_item["news_id"], news_item["title"], news_item["full_text"])
                incoming_news.append(news_tuple)
            except KeyError as e:
                self.logger.warning(f"Agent {self.id} skipped invalid news item due to missing key: {e}")
                continue

        await self.memory.status.update("incoming_news_buffer", incoming_news)

        # Step 2: Immediately select news from the buffer (process every time a news arrives)
        selected_news = await self.select_news_from_batch(incoming_news, reflection_batch_size)

        # Add delay after LLM/API call to prevent hitting rate limits (adjust as needed)
        

        # Clear the incoming buffer after selection
        await self.memory.status.update("incoming_news_buffer", [])

        if not selected_news:
            return ""


        # Log which articles were added
        selected_ids = [news[0] for news in selected_news]
        self.logger.info(f"Agent {self.id} selected news {selected_ids} and added to unread memory.")

        # Step 3: Log which articles were selected
        selected_ids = [news[0] for news in selected_news]
        self.logger.info(f"Agent {self.id} selected news {selected_ids} and will immediately reflect on them.")

        # Update the master list of selected IDs (optional)
        all_selected_ids = await self.memory.status.get("selected_news_ids")
        all_selected_ids.extend(selected_ids)
        await self.memory.status.update("selected_news_ids", all_selected_ids)

        # Step 4: Immediately reflect on this batch (no threshold)
        await self.self_reflection_on_batch(selected_news)

        # After reflection:
        all_selected_ids = await self.memory.status.get("selected_news_ids")
        # Remove the IDs that were just reflected upon
        remaining_ids = [id for id in all_selected_ids if id not in selected_ids]
        await self.memory.status.update("selected_news_ids", remaining_ids)
        return ""


    async def pretty_profile_section(self, profile_dict):
        """Return a readable string for a user profile subdict."""
    
        PROFILE_FIELD_DESCRIPTIONS = {
            # Demographics
            "sex": "Sex",
            "race": "Race/ethnicity",
            "marital": "Marital status",
            "degree": "Highest degree earned",
            "educ": "Years of education",
            "region": "Region",
            "income": "Income",
            "class": "Self-identified social class",
            "age": "Age",
            "partyid": "Political party identification",
            "polviews": "Political views",
            "leftrght": "Left/right political orientation",
            # Media Habits
            "polnews": "Frequency of following political news",
            "polnewsfrom": "Sources of political news",
            "smnews": "Frequency of getting news from social media",
            "tvnews1": "Frequency of watching news on TV",
            "papernews": "Frequency of reading news in newspapers",
            "radionews": "Frequency of listening to news on the radio",
            "webnews": "Frequency of reading news online",
            # Domestic Views, etc.
            "lkelyvot": "Likelihood to vote",
            "polintrst": "Interest in politics",
            "cantrust": "Trust in other people",
            "befair": "Belief that people are generally fair",
            "corruptn": "Perception of corruption in government",
            "oppsegov": "Opinion on size/scope of government",
            "rghtsmin": "Support for rights of minorities",
            "fulldem": "Belief whether the country is a full democracy",
            "eqwlth": "Attitude toward equality of wealth",
            "tax": "Attitude toward taxes",
            "gunlaw": "Support for stricter gun laws",
            "cappun": "Support for capital punishment",
            "nataid": "Attitude toward foreign aid",
            "natspac": "Support for spending on space exploration",
            "natenvir": "Support for spending on the environment",
            "natheal": "Support for spending on health",
            "imports": "Attitude toward imports and trade",
            "immcrime": "Belief about immigrants and crime",
            "immjobs": "Belief about immigrants and jobs",
            "immameco": "Belief about immigrants and the economy",
            "union1": "Membership in a labor union",
            "satfin": "Satisfaction with personal/household finances",
            "grass": "Opinion on legalization of marijuana",
            "conlegis": "Confidence in the legislative branch",
            "conjudge": "Confidence in the judiciary/courts",
            "conarmy": "Confidence in the military",
            "wrldgovt": "Opinion on world government",
        }

        lines = []
        for k, v in profile_dict.items():
            desc = PROFILE_FIELD_DESCRIPTIONS.get(k, k)
            # Skip if value is None or empty
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            lines.append(f"- {desc}: {v}")
        return "\n".join(lines) if lines else "None reported"

    async def select_news_from_batch(
        self, 
        news_batch: list[tuple[str, str, str]], 
        reflection_batch_size: int = 25,
    ) -> list[tuple[str, str]]:
        """
        Selects articles to read from a batch of news articles by asking the LLM to
        extract the specific news_id.

        Args:
            news_batch: List of (news_id, title, full_text) tuples.

        Returns:
            List of 5 (title, full_text) tuples that the agent wants to read.
        """
        
        #debug what the news batch contains
        # get_logger().info(f"Agent {self.id} received news batch for selection: {news_batch}")
        # test whether reflection_batch_size is set correctly
        # self.logger.info(f"Agent {self.id} using reflection_batch_size: {reflection_batch_size}") 
        profile = await self.memory.status.get("profile") or {}
        
        interests = profile.get("interests", [])
        china_attitude = await self.memory.status.get("china_attitude", "neutral")

        # --- Format the news as a list of JSON objects for the prompt ---
        news_as_json_list = []
        for news_id, title, _ in news_batch:
            # Create a mini JSON object for each article
            news_as_json_list.append(json.dumps({"id": news_id, "title": title}))
        
        formatted_news_list = "\n".join(news_as_json_list)

        demographics = self.demographics 
        interests = self.interests 
        domestic_views = self.domestic_views 
        media_habits = self.media_habits
        
        demographics_str = await self.pretty_profile_section(demographics)
        media_habits_str = await self.pretty_profile_section(media_habits)
        domestic_views_str = await self.pretty_profile_section(domestic_views)

        # get_logger().info(f"Agent {self.id} formatted news list for selection: {formatted_news_list}")

        # --- Create a much more specific prompt with an example ---
        selection_prompt = f"""
        You are a data selection assistant. Your task is to analyze a list of news articles and select the ones that are most relevant to a user's profile.

        **User Profile:**
        - Demographics: {demographics_str}
        - Interests: {interests}
        - Attitude toward China: {china_attitude}
        - Media Habits: {media_habits_str}

        **News Articles (List of JSON Objects):**
        {formatted_news_list}

        **Instructions:**
        1. Review the user profile and the list of news articles.
        2. Select exactly **{reflection_batch_size}** articles that are most interesting to the user.
        3. Your output MUST be a single JSON object containing a list of the selected article IDs.
        4. For each article you select, you must find its corresponding JSON object from the list above and **copy the value of the 'id' field exactly as it appears.** Do not shorten or change the ID.

        **EXAMPLE:**
        If the input list contains:
        {{"id": "12376", "title": "A highly relevant title about politics"}}
        {{"id": "44", "title": "An irrelevant title about sports"}}
        {{"id": "523537", "title": "Another relevant title about the economy"}}

        And you choose the first and second articles, your output MUST be formatted exactly like this:
        {{"selected_ids": ["12376", "44"]}}
        """
        
        try:
            async with llm_semaphore:
                await asyncio.sleep(getattr(self, "llm_delay", 0.25))
                response = await self.llm.atext_request(
                    dialog=[
                        {"role": "system", "content": "You are an assistant that helps select news articles and returns data in a strict JSON format."},
                        {"role": "user", "content": selection_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                # Debug: Log the LLM response
                get_logger().info(f"Agent {self.id} received response for news selection: {response}")
                selected_ids_data = json_repair.loads(response)
                
                # Ensure selected_ids contains unique IDs
                selected_ids = set(selected_ids_data.get("selected_ids", []))
                get_logger().info(f"Agent {self.id} extracted selected_ids: {selected_ids}")

                # Ensure selected_ids contains exactly 5 IDs
                if len(selected_ids) != reflection_batch_size:
                    self.logger.warning(f"Agent {self.id} selected {len(selected_ids)} articles instead of {reflection_batch_size}.")
        except Exception as e:
            self.logger.error(f"Agent {self.id} failed to select news from batch: {e}")
            return []

        # Match selected IDs with the news batch
        selected_news = [
            (title, full_text)
            for news_id, title, full_text in news_batch
            if news_id in selected_ids
        ]
        
        # Log the selected titles
        selected_titles = [title for title, _ in selected_news]
        # self.logger.info(f"Agent {self.id} 从第{batch_number}个20条新闻中选择了5篇阅读: {selected_titles}")
        return selected_news



    async def self_reflection_on_batch(self, news_batch: List[Tuple[str, str, str]]):
        """
        Core self-reflection on a BATCH of news to update opinions.
        """
        self.logger.info(f"Agent {self.id} starting batch self-reflection on {len(news_batch)} articles.")

        current_opinions = await self.memory.status.get("china_opinions")
        profile_data = await self.memory.status.get("profile") or {}
        
        # Propose updates based on the entire batch
        #log the news batch
        # self.logger.info(f"Agent {self.id} received batch for reflection: {news_batch}")
        proposed_updates_data = await self.reason_about_news_batch(news_batch, current_opinions, profile_data)
        proposed_updates = proposed_updates_data.get("updates", [])

        if not proposed_updates:
            self.logger.info(f"Batch reflection resulted in no opinion changes.")
            return

        # Apply updates (this logic is similar to before but logs are batch-oriented)
        updated_domains = []
        for update in proposed_updates:
            domain = update.get("domain")
            if domain in self.OPINION_DOMAINS:
                old_valence = current_opinions[domain]["valence"]
                current_opinions[domain]["valence"] = float(update.get("new_valence", old_valence))
                # Increment exposure for each domain mentioned in the batch reflection
                current_opinions[domain]["exposure_count"] += 1
                updated_domains.append(domain)
                self.logger.info(
                    f"Agent {self.id} [BATCH UPDATE] Opinion for {domain}: "
                    f"old={old_valence:.2f}, new={current_opinions[domain]['valence']:.2f}, "
                    f"reasoning='{update.get('reasoning', 'N/A')}'"
                )
        
        # Persist changes and update overall attitude
        if updated_domains:
            await self.memory.status.update("china_opinions", current_opinions)
            await self.update_overall_attitude(current_opinions)
            self.logger.info(f"Agent {self.id} completed batch reflection. Updated domains: {', '.join(updated_domains)}")
            # You might want to log the entire reflection reasoning to the stream memory
            await self.memory.stream.add(topic="batch_self_reflection", description=json.dumps(proposed_updates_data))


    async def reason_about_news_batch(self, news_batch: List[Tuple[str, str]], current_opinions: dict, profile: dict) -> dict:
        """
        Performs reasoning on a batch of news to propose opinion updates.
        """
        # Format the batch of news for the prompt
        # formatted_news_list = "\n".join([f"- Title: {title}" for title, _ in news_batch])


        MAX_CHARS = 200
        formatted_news_list = "\n".join([
            f"- Title: {title}\n  Full Text: {full_text[:MAX_CHARS]}{'...' if len(full_text) > MAX_CHARS else ''}"
            for title, full_text in news_batch
        ])
        
        demographics = self.demographics 
        interests = self.interests 
        domestic_views = self.domestic_views 
        media_habits = self.media_habits
        
        demographics_str = await self.pretty_profile_section(demographics)
        media_habits_str = await self.pretty_profile_section(media_habits)
        domestic_views_str = await self.pretty_profile_section(domestic_views)
        # UPDATED Prompt for Batch Reflection
        reasoning_prompt = f"""
        You are the reasoning core of an AI agent. Your task is to reflect on a batch of news articles you've just read and update your opinions about China.

        **Your Current Opinions:**
        ```json
        {json.dumps(current_opinions, indent=2)}
        ```

        **Your Personal Profile:**
        - Interests: {', '.join(interests) if interests else 'No specific interests listed'}
        - Demographics: {demographics_str}
        - Domestic Views: {domestic_views_str}

        

        **The New Information (A Batch of {len(news_batch)} News Articles):**
        {formatted_news_list}

        **Your Task: Perform Holistic Self-Reflection on the ENTIRE BATCH**
        Synthesize the information from all the articles at once. Do not go one-by-one.
        1. **Identify Key Themes & Relevant Domains:** What are the overarching topics across these articles (e.g., economic tension, technological collaboration)? Which opinion domains do they touch upon?
        2. **Consider Your Profile:** Reflect on how your personal interests and characteristics (political interest level, trust in people, views on corruption, and wealth equality attitudes) might make certain aspects of the news more or less significant to you.
        3. **Reason About the Impact:** For each relevant domain, consider how the batch of news should change your opinion based on your personal profile.
        4. **Compare new information with existing beliefs**
        - First summarize the polarity of your views on the categories covered in this article.
        - Then compare the new information with your existing beliefs. Does it confirm or contradict your views?

        If it confirms, then your opinion should become stronger or stay the same. If it contradicts, then you should choose from among the following:
        - Change the Dissonant Cognition: Altering your existing belief to make them more consistent with the new information.
        - Adding a New Cognition: Introducing a new thoughts or belief, via reasoning, that will help to justify or strengthen your existing view despite it being contradictory to the information.
        - Reducing the Importance of the Dissonant Cognition: De-emphasizing the significance of conflicting beliefs or behaviors.

        5. Propose Updates: Based on your reasoning, determine the `new_valence` for each affected domain. You must provide a `new_valence`, and you cannot keep the same value as before. The `new_valence` should be a float between -2.0 and 2.0, where:
        
        **Output Format:**
        Return a JSON object with your reasoning and proposed updates.
        {{
          "synthesis_reasoning": "A brief summary of how you interpreted the batch of news as a whole, considering your profile and how you reconciled any conflicting information.",
          "updates": [
            {{
              "domain": "economy_business",
              "new_valence": -1.2,
              "reasoning": "The overall negative tone of several articles about trade disputes outweighs the single positive one, shifting my opinion negatively."
            }},
            {{
              "domain": "technology_innovation",
              "new_valence": 1.5,
              "reasoning": "Despite some economic friction, the consistent news of tech breakthroughs strongly reinforces my positive view in this specific area."
            }}
          ]
        }}
        """
        
        try:
            async with llm_semaphore:
                await asyncio.sleep(getattr(self, "llm_delay", 0.25)) 
                response = await self.llm.atext_request(
                    dialog=[
                        {"role": "system", "content": "You are a reasoning engine for an agent's self-reflection on a batch of news."},
                        {"role": "user", "content": reasoning_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                return json_repair.loads(response)
        except Exception as e:
            self.logger.warning(f"Batch reasoning and update proposal failed: {e}")
            return {"updates": []}

    async def update_overall_attitude(self, domain_opinions: dict):
        """
        Update the overall attitude toward China based on weighted domain opinions.
        Uses exposure_count as weights - domains with more exposure have more influence.
        """
        total_weighted_valence = 0.0
        total_weight = 0.0
        
        for domain, opinion in domain_opinions.items():
            valence = opinion.get("valence", 0.0)
            exposure_count = opinion.get("exposure_count", 0)
            
            # Use exposure_count as weight
            weight = exposure_count
            
            total_weighted_valence += valence * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_valence = total_weighted_valence / total_weight
        else:
            # If no exposure yet, default to neutral
            overall_valence = 0.0
            
        # Convert to descriptive attitude
        if overall_valence >= 1.0:
            attitude = "very positive"
        elif overall_valence >= 0.25:
            attitude = "positive"
        elif overall_valence > -0.25:
            attitude = "neutral"
        elif overall_valence > -1.0:
            attitude = "negative"
        else:
            attitude = "very negative"
        
        await self.memory.status.update("china_attitude", attitude)


    
