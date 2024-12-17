import logging
from typing import Optional

from aixplain.factories import ModelFactory
from embedchain.config import BaseLlmConfig
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.llm.base import BaseLlm

logger = logging.getLogger(__name__)

@register_deserializable
class AIXplainEmbedChainLL(BaseLlm):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        """
        Initialize the AIXplainEmbedChainLL class.

        :param config: LLM configuration option class, defaults to None
        :type config: Optional[BaseLlmConfig], optional
        """
        super().__init__(config=config)
        
        self.model_id = config.model if config and config.model else "66b270bb6eb56365551e8c71"
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the model using aixplain ModelFactory.
        """
        try:
            self.model = ModelFactory.get(self.model_id)
            logger.info(f"Model initialized with ID: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise ValueError("Could not initialize the AIXplain model.")

    def get_llm_model_answer(self, prompt):
        """
        Get the answer from the AIXplain model based on the given prompt.

        :param prompt: The input prompt to the model
        :type prompt: str
        :return: The model's response
        :rtype: str
        """
        if not self.model:
            raise ValueError("Model is not initialized. Make sure the model ID is valid.")

        try:
            # Prepare the input payload for the model.
            payload = {
                "text": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            
            }

            # Filter out None values to avoid API issues.
            payload = {key: value for key, value in payload.items() if value is not None}

            # Run the model and get the result.
            result = self.model.run(payload)

            if "data" in result:
                return result["data"]
            else:
                logger.error("Unexpected response format from the model.")
                raise ValueError("Model response did not contain the expected 'result' field.")

        except Exception as e:
            logger.error(f"Failed to get response from the model: {e}")
            raise RuntimeError("Error occurred while querying the AIXplain model.")
        
        



import logging
from typing import Optional

from embedchain.config import BaseEmbedderConfig
from embedchain.embedder.base import BaseEmbedder
from embedchain.models import VectorDimensions
from aixplain.factories import ModelFactory
from embedchain.embedder.base import EmbeddingFunc


logger = logging.getLogger(__name__)

class AixplainEmbedder(BaseEmbedder):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        """
        Initialize the Aixplain embedder.

        :param config: Optional configuration for the embedder.
        """
        super().__init__(config=config)

        if self.config.model is None:
            self.config.model = "6734c55df127847059324d9e"  

        # Initialize the Aixplain model
        try:
            self.model = ModelFactory.get(self.config.model)
            # logger.info(f"Aixplain embedding model initialized with ID: {self.config["model"]})
        except Exception as e:
            logger.error(f"Failed to initialize Aixplain model: {e}")
            raise ValueError("Could not initialize the Aixplain embedding model.")

        
        vector_dimension = self.config.vector_dimension or 1536
        self.set_vector_dimension(vector_dimension=vector_dimension)
        embed_f = EmbeddingFunc(self.embed) 
        self.set_embedding_fn(embedding_fn=embed_f)

    def embed(self, text: list[str]) -> list[float]:
        """
        Generate embeddings for the provided text using Aixplain.

        :param text: Input text to embed.
        :return: List of embedding values.
        """
        try:
            logger.info(f"Generating embeddings for text: {text}")
            result = self.model.run(text)
            print("Generated embeddings: ", result )

            if result and 'data' in result and isinstance(result['data'], list):
                embedding = [result['data'][i]['embedding'] for i in range(len(result["data"]))]
                logger.info(f"Generated embeddings successfully.")
                return embedding

            logger.error("Unexpected response format from Aixplain model.")
            raise ValueError("Unexpected response format from Aixplain embedding model.")

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError("Error occurred while generating embeddings.")





import ast
import concurrent.futures
import json
import logging
import os
from typing import Any, Optional, Union

import requests
import yaml
from tqdm import tqdm
from embedchain import App
from embedchain.cache import (
    Config,
    ExactMatchEvaluation,
    SearchDistanceEvaluation,
    cache,
    gptcache_data_manager,
    gptcache_pre_function,
)
from embedchain.client import Client
from embedchain.config import AppConfig, CacheConfig, ChunkerConfig, Mem0Config
from embedchain.core.db.database import get_session
from embedchain.core.db.models import DataSource
from embedchain.embedchain import EmbedChain
from embedchain.embedder.base import BaseEmbedder
from embedchain.embedder.openai import OpenAIEmbedder
from embedchain.evaluation.base import BaseMetric
from embedchain.evaluation.metrics import (
    AnswerRelevance,
    ContextRelevance,
    Groundedness,
)
from embedchain.factory import EmbedderFactory, LlmFactory, VectorDBFactory
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.llm.base import BaseLlm
from embedchain.llm.openai import OpenAILlm
from embedchain.telemetry.posthog import AnonymousTelemetry
from embedchain.utils.evaluation import EvalData, EvalMetric
from embedchain.utils.misc import validate_config
from embedchain.vectordb.base import BaseVectorDB
from embedchain.vectordb.chroma import ChromaDB
from mem0 import Memory

logger = logging.getLogger(__name__)

class AixplainEmbedChainApp(App):
    """
    Custom App implementation using Aixplain models.
    """
    
    @classmethod
    def from_config(
        cls,
        config_path: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        auto_deploy: bool = False,
        yaml_path: Optional[str] = None,
    ):
        """
        Instantiate a App object from a configuration.

        :param config_path: Path to the YAML or JSON configuration file.
        :type config_path: Optional[str]
        :param config: A dictionary containing the configuration.
        :type config: Optional[dict[str, Any]]
        :param auto_deploy: Whether to deploy the app automatically, defaults to False
        :type auto_deploy: bool, optional
        :param yaml_path: (Deprecated) Path to the YAML configuration file. Use config_path instead.
        :type yaml_path: Optional[str]
        :return: An instance of the App class.
        :rtype: App
        """
        # Backward compatibility for yaml_path
        if yaml_path and not config_path:
            config_path = yaml_path

        if config_path and config:
            raise ValueError("Please provide only one of config_path or config.")

        config_data = None

        if config_path:
            file_extension = os.path.splitext(config_path)[1]
            with open(config_path, "r", encoding="UTF-8") as file:
                if file_extension in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(file)
                elif file_extension == ".json":
                    config_data = json.load(file)
                else:
                    raise ValueError("config_path must be a path to a YAML or JSON file.")
        elif config and isinstance(config, dict):
            config_data = config
        else:
            logger.error(
                "Please provide either a config file path (YAML or JSON) or a config dictionary. Falling back to defaults because no config is provided.",  # noqa: E501
            )
            config_data = {}

        # Validate the config
        # validate_config(config_data)

        app_config_data = config_data.get("app", {}).get("config", {})
        vector_db_config_data = config_data.get("vectordb", {})
        embedding_model_config_data = config_data.get("embedding_model", config_data.get("embedder", {}))
        memory_config_data = config_data.get("memory", {})
        llm_config_data = config_data.get("llm", {})
        chunker_config_data = config_data.get("chunker", {})
        cache_config_data = config_data.get("cache", None)

        app_config = AppConfig(**app_config_data)
        memory_config = Mem0Config(**memory_config_data) if memory_config_data else None

        vector_db_provider = vector_db_config_data.get("provider", "chroma")
        vector_db = VectorDBFactory.create(vector_db_provider, vector_db_config_data.get("config", {}))

        if llm_config_data:
            llm_provider = llm_config_data.get("provider", "openai")
            if llm_provider =="aixplain":
                llm = AIXplainEmbedChainLL(BaseLlmConfig(**llm_config_data.get("config", {})))
            else:
                llm = LlmFactory.create(llm_provider, llm_config_data.get("config", {}))
                    
            
        else:
            llm = None
        
        embedding_model_provider = embedding_model_config_data.get("provider", "openai")
        if embedding_model_provider == "aixplain":
            embedding_model = AixplainEmbedder(BaseEmbedderConfig(**embedding_model_config_data.get("config", {})))
        else:
            embedding_model = EmbedderFactory.create(
            embedding_model_provider, embedding_model_config_data.get("config", {})
        )
                
        if cache_config_data is not None:
            cache_config = CacheConfig.from_config(cache_config_data)
        else:
            cache_config = None
        
        
        config_data  =dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
        
        return cls(
            config=app_config,
            llm=llm,
            db=vector_db,
            embedding_model=embedding_model,
            config_data=config_data,
            auto_deploy=auto_deploy,
            chunker=chunker_config_data,
            cache_config=cache_config,
            memory_config=memory_config,
        )
        


if __name__ == "__main__":
    
    config = BaseLlmConfig(temperature= 0.7,max_tokens= 265, top_p = 0.5)
    
    model = AIXplainEmbedChainLL(config=config)
    
    
    
    model.get_llm_model_answer("Hello, How are you ?")
    
    
    
    
    

    