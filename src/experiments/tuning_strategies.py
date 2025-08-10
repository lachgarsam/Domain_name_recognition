import logging
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PrefixTuningConfig, IA3Config, TaskType, PeftConfig

from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TuningStrategy(ABC):
    """
    Abstract base class for different model tuning strategies using PEFT.
    """

    def __init__(self, strategy_config: Dict[str, Any]) -> None:
        """
        Initialize the tuning strategy with its specific configuration.

        Parameters:
            strategy_config (Dict[str, Any]): Configuration dictionary for the tuning strategy.
        """
        self.config = strategy_config

    @abstractmethod
    def prepare_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Prepare or modify the model before applying PEFT tuning.

        Parameters:
            model (AutoModelForCausalLM): The pre-trained causal language model.

        Returns:
            AutoModelForCausalLM: The prepared or modified model.
        """
        pass

    @abstractmethod
    def get_peft_config(self) -> Optional[PeftConfig]:
        """
        Return the PEFT configuration specific to this tuning strategy.

        Returns:
            Optional[PeftConfig]: PEFT configuration object or None if full tuning is used.
        """
        pass


class FullTuningStrategy(TuningStrategy):
    """
    Full fine-tuning strategy without any PEFT modifications.
    """

    def prepare_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        logger.info(
            "Applying Full Fine-Tuning strategy. No model modifications needed."
        )
        return model

    def get_peft_config(self) -> PeftConfig | None:
        logger.info("Full Fine-Tuning selected. No PEFT config will be used.")
        return None


class LoRAStrategy(TuningStrategy):
    """
    LoRA (Low-Rank Adaptation) tuning strategy.
    """

    def prepare_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        logger.info("Applying LoRA strategy. No model modifications needed.")
        return model

    def get_peft_config(self) -> PeftConfig:
        logger.info(f"Creating LoraConfig with params: {self.config}")
        return LoraConfig(**self.config, task_type=TaskType.CAUSAL_LM)


class QLoRAStrategy(TuningStrategy):
    """
    QLoRA tuning strategy with quantization handled at model load time.
    """

    def prepare_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        logger.info(
            "QLoRA strategy selected. Quantization is handled at model load time."
        )
        return model

    def get_peft_config(self) -> PeftConfig:
        lora_params = self.config["lora"]
        logger.info(f"Creating LoraConfig for QLoRA with params: {lora_params}")
        return LoraConfig(**lora_params, task_type=TaskType.CAUSAL_LM)


class PrefixTuningStrategy(TuningStrategy):
    """
    Prefix-tuning strategy.
    """

    def prepare_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        logger.info("Applying Prefix-Tuning strategy.")
        return model

    def get_peft_config(self) -> PeftConfig:
        logger.info(f"Creating PrefixTuningConfig with params: {self.config}")
        return PrefixTuningConfig(**self.config, task_type=TaskType.CAUSAL_LM)


class IA3Strategy(TuningStrategy):
    """
    IA³ tuning strategy.
    """

    def prepare_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        logger.info("Applying IA³ strategy. No model modifications needed.")
        return model

    def get_peft_config(self) -> PeftConfig:
        """Crée et retourne une configuration IA3Config."""
        logger.info(f"Creating IA3Config with params: {self.config}")
        return IA3Config(**self.config, task_type=TaskType.CAUSAL_LM)


def get_tuning_strategy(config: Dict[str, Any]) -> TuningStrategy:
    """
    Factory function to select and instantiate the correct tuning strategy based on config.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary containing the 'tuning' key.

    Returns:
        TuningStrategy: An instance of a tuning strategy class.

    Raises:
        ValueError: If no valid tuning strategy is found in the configuration.
    """
    tuning_config = config["tuning"]

    if "full_tuning" in tuning_config:
        return FullTuningStrategy(tuning_config["full_tuning"])
    if "lora" in tuning_config:
        return LoRAStrategy(tuning_config["lora"])
    if "qlora" in tuning_config:
        return QLoRAStrategy(tuning_config["qlora"])
    if "prefix_tuning" in tuning_config:
        return PrefixTuningStrategy(tuning_config["prefix_tuning"])

    if "ia3" in tuning_config:
        return IA3Strategy(tuning_config["ia3"])

    raise ValueError(
        "No valid tuning strategy found in the configuration under the 'tuning' key."
    )
