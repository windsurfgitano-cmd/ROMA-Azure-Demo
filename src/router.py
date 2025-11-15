"""
Router Dinámico Inteligente para ROMA con Azure AI Foundry
Selecciona automáticamente el modelo óptimo según contexto
"""

import dspy
from typing import Literal, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TaskComplexity(Enum):
    """Niveles de complejidad"""
    ULTRA = "ultra"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"


class TaskPriority(Enum):
    """Prioridades de ejecución"""
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"
    COST = "cost"
    REASONING = "reasoning"


class TaskDomain(Enum):
    """Dominios especializados"""
    CODE = "code"
    DOCUMENT = "document"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    GENERAL = "general"
    REALTIME = "realtime"


@dataclass
class ModelConfig:
    """Configuración de modelo"""
    name: str
    provider: str
    temperature: float
    cache: bool = True
    max_tokens: Optional[int] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None


class IntelligentRouter:
    """Router inteligente de modelos"""

    def __init__(self, azure_config: Optional[Dict[str, str]] = None):
        self.azure_config = azure_config or {}
        self._model_cache: Dict[str, dspy.LM] = {}

        # Registro de modelos disponibles
        self.model_registry = {
            # Reasoning models
            "gpt5-chat": ModelConfig(
                name="gpt-5-chat",
                provider="azure",
                temperature=1.0,
                max_tokens=16000,
                **self.azure_config
            ),
            "deepseek-r1": ModelConfig(
                name="DeepSeek-R1-0528",
                provider="azure",
                temperature=0.4,
                max_tokens=250_000,
                **self.azure_config
            ),

            # Fast models
            "gpt4o": ModelConfig(
                name="gpt-4o",
                provider="azure",
                temperature=0.7,
                **self.azure_config
            ),
            "grok-fast": ModelConfig(
                name="grok-4-fast-reasoning",
                provider="azure",
                temperature=0.6,
                max_tokens=250_000,
                **self.azure_config
            ),

            # Specialized
            "codestral": ModelConfig(
                name="Codestral-2501",
                provider="azure",
                temperature=0.3,
                **self.azure_config
            ),
        }

        # Routing matrix
        self.routing_matrix = self._build_routing_matrix()

    def _build_routing_matrix(self) -> Dict:
        """Matriz de decisión de routing"""
        return {
            TaskComplexity.ULTRA: {
                TaskPriority.QUALITY: {
                    TaskDomain.CODE: "codestral",
                    TaskDomain.ANALYSIS: "gpt5-chat",
                    TaskDomain.GENERAL: "gpt5-chat",
                },
                TaskPriority.REASONING: {
                    TaskDomain.CODE: "deepseek-r1",
                    TaskDomain.ANALYSIS: "deepseek-r1",
                    TaskDomain.GENERAL: "gpt5-chat",
                },
            },
            TaskComplexity.HIGH: {
                TaskPriority.QUALITY: {
                    TaskDomain.CODE: "codestral",
                    TaskDomain.ANALYSIS: "gpt4o",
                    TaskDomain.GENERAL: "gpt4o",
                },
                TaskPriority.SPEED: {
                    TaskDomain.GENERAL: "grok-fast",
                },
            },
            TaskComplexity.MEDIUM: {
                TaskPriority.BALANCED: {
                    TaskDomain.GENERAL: "gpt4o",
                },
            },
        }

    def get_model(
        self,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        priority: TaskPriority = TaskPriority.BALANCED,
        domain: TaskDomain = TaskDomain.GENERAL,
    ) -> dspy.LM:
        """Obtiene modelo óptimo"""

        # Buscar en matriz
        try:
            model_key = self.routing_matrix[complexity][priority].get(
                domain,
                self.routing_matrix[complexity][priority].get(TaskDomain.GENERAL, "gpt4o")
            )
        except KeyError:
            model_key = "gpt4o"

        # Cache
        if model_key in self._model_cache:
            return self._model_cache[model_key]

        # Crear instancia
        config = self.model_registry[model_key]

        lm = dspy.LM(
            f"{config.provider}/{config.name}",
            temperature=config.temperature,
            cache=config.cache,
            api_base=config.api_base,
            api_key=config.api_key,
            api_version=config.api_version,
            **({"max_tokens": config.max_tokens} if config.max_tokens else {})
        )

        self._model_cache[model_key] = lm
        return lm


# Ejemplo de uso
if __name__ == "__main__":
    azure_config = {
        "api_base": "https://YOUR-RESOURCE.openai.azure.com",
        "api_key": "YOUR_API_KEY",
        "api_version": "2025-01-01-preview"
    }

    router = IntelligentRouter(azure_config=azure_config)

    # Obtener modelo para tarea compleja
    model = router.get_model(
        complexity=TaskComplexity.ULTRA,
        priority=TaskPriority.REASONING,
        domain=TaskDomain.ANALYSIS
    )

    print(f"Selected model: {model}")
