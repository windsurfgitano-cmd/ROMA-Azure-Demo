"""
Demo completo de ROMA con Azure AI Foundry
Ejecuta un workflow completo: Atomizer -> Planner -> Executor -> Aggregator -> Verifier
"""

import os
import dspy
from typing import Dict, Any
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de Azure
AZURE_CONFIG = {
    "api_base": os.getenv("AZURE_API_BASE"),
    "api_key": os.getenv("AZURE_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
}


# ============================================
# Signatures ROMA
# ============================================

class AtomizerSignature(dspy.Signature):
    """Determina si una tarea es atómica o requiere descomposición"""
    goal = dspy.InputField(desc="La tarea o pregunta a analizar")
    is_atomic = dspy.OutputField(desc="True si puede resolverse directamente, False si requiere planificación")
    reasoning = dspy.OutputField(desc="Explicación de la decisión")


class PlannerSignature(dspy.Signature):
    """Descompone tareas complejas en subtareas"""
    goal = dspy.InputField(desc="Objetivo complejo a descomponer")
    subtasks = dspy.OutputField(desc="Lista de subtareas específicas")
    strategy = dspy.OutputField(desc="Estrategia general")


class ExecutorSignature(dspy.Signature):
    """Ejecuta tareas atómicas"""
    task = dspy.InputField(desc="Tarea específica a ejecutar")
    result = dspy.OutputField(desc="Resultado detallado")


class AggregatorSignature(dspy.Signature):
    """Sintetiza múltiples resultados"""
    original_goal = dspy.InputField(desc="Objetivo original")
    subtask_results = dspy.InputField(desc="Resultados de subtareas")
    synthesized_result = dspy.OutputField(desc="Respuesta final integrada")


class VerifierSignature(dspy.Signature):
    """Verifica calidad del resultado"""
    goal = dspy.InputField(desc="Objetivo original")
    result = dspy.InputField(desc="Resultado a verificar")
    is_valid = dspy.OutputField(desc="True si cumple")
    feedback = dspy.OutputField(desc="Feedback")


# ============================================
# Pipeline ROMA
# ============================================

def solve_with_roma(goal: str) -> Dict[str, Any]:
    """Pipeline completo ROMA"""

    print(f"\n{'='*70}")
    print(f"🎯 OBJETIVO: {goal}")
    print(f"{'='*70}\n")

    # Configurar modelos
    gpt5_chat = dspy.LM(
        model="azure/gpt-5-chat",
        api_base=AZURE_CONFIG["api_base"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        temperature=1.0,
        max_tokens=16000,
        cache=True
    )

    gpt4o = dspy.LM(
        model="azure/gpt-4o",
        api_base=AZURE_CONFIG["api_base"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        temperature=0.7,
        cache=True
    )

    # Crear módulos
    atomizer = dspy.ChainOfThought(AtomizerSignature)
    planner = dspy.ChainOfThought(PlannerSignature)
    executor = dspy.ChainOfThought(ExecutorSignature)
    aggregator = dspy.ChainOfThought(AggregatorSignature)
    verifier = dspy.Predict(VerifierSignature)

    results = {"goal": goal, "steps": []}

    # PASO 1: Atomizer
    print("[1/5] 🔍 ATOMIZER...")
    with dspy.context(lm=gpt4o):
        atomized = atomizer(goal=goal)
        is_atomic = "true" in atomized.is_atomic.lower()
        print(f"   └─ Es atómica: {is_atomic}")
        results["steps"].append({"module": "atomizer", "is_atomic": is_atomic})

    if is_atomic:
        # Ejecución directa
        print("[2/5] ⚙️  EXECUTOR...")
        with dspy.context(lm=gpt5_chat):
            execution = executor(task=goal)
            final_result = execution.result
            results["steps"].append({"module": "executor", "result": final_result})
    else:
        # Descomposición
        print("[2/5] 📋 PLANNER...")
        with dspy.context(lm=gpt5_chat):
            plan = planner(goal=goal)
            results["steps"].append({"module": "planner", "strategy": plan.strategy})

        print("[3/5] ⚙️  EXECUTOR (subtareas)...")
        subtask_results = []
        sample_subtasks = [
            "Investigar información relevante",
            "Analizar datos y tendencias",
            "Generar conclusiones"
        ]

        for i, subtask in enumerate(sample_subtasks, 1):
            print(f"   [{i}/3] {subtask[:50]}...")
            with dspy.context(lm=gpt4o):
                sub_exec = executor(task=f"{goal} - {subtask}")
                subtask_results.append({"task": subtask, "result": sub_exec.result})

        print("[4/5] 🔀 AGGREGATOR...")
        formatted_results = "\n\n".join([
            f"Subtarea {i+1}: {r['task']}\nResultado: {r['result'][:200]}..."
            for i, r in enumerate(subtask_results)
        ])

        with dspy.context(lm=gpt5_chat):
            aggregated = aggregator(
                original_goal=goal,
                subtask_results=formatted_results
            )
            final_result = aggregated.synthesized_result

    # PASO 5: Verifier
    print("[5/5] ✅ VERIFIER...")
    with dspy.context(lm=gpt4o):
        verification = verifier(goal=goal, result=final_result[:1000])
        is_valid = "true" in verification.is_valid.lower()
        print(f"   └─ Válido: {is_valid}")

    results["final_result"] = final_result
    results["is_valid"] = is_valid

    return results


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    # Verificar configuración
    if not AZURE_CONFIG["api_base"] or not AZURE_CONFIG["api_key"]:
        print("❌ Error: Configura tus credenciales de Azure en .env")
        exit(1)

    # Ejecutar demo
    goal = "Explica las 3 tendencias más importantes en IA para 2025"

    try:
        result = solve_with_roma(goal)

        print(f"\n{'='*70}")
        print("✨ RESULTADO FINAL")
        print(f"{'='*70}")
        print(f"\n{result['final_result']}\n")
        print(f"✅ Válido: {result['is_valid']}")
        print(f"📊 Pasos ejecutados: {len(result['steps'])}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
