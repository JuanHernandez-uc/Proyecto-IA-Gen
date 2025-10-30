import argparse
import json
import torch
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
from huggingface_hub import login
import os

login(token="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")
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def create_judge_prompt(generated_answer, respuestas_correctas, respuestas_aliases):
    valid_answers = []
    valid_answers.extend(respuestas_correctas)

    for alias_group in respuestas_aliases:
        if isinstance(alias_group, list):
            valid_answers.extend(alias_group)
        elif isinstance(alias_group, str):
            valid_answers.append(alias_group)

    # Limpiar y deduplicar
    valid_answers = [str(a).strip() for a in valid_answers if a]
    valid_answers = list(set(valid_answers))

    prompt = f"""Eres un evaluador experto. Tu tarea es determinar si dos respuestas son semánticamente equivalentes o suficientemente similares, incluso si están escritas de forma diferente, en diferentes idiomas, o con diferentes niveles de especificidad.

Respuesta generada por el modelo: "{generated_answer}"

Respuestas correctas aceptadas: {', '.join(valid_answers)}

Analiza si la respuesta generada es correcta o suficientemente similar a alguna de las respuestas aceptadas. Considera:
- Sinónimos y traducciones (ej: "fútbol" y "football" son lo mismo)
- Variaciones de escritura (ej: "béisbol" y "beisbol")
- Niveles de especificidad (ej: "balonmano" es similar a "balonmano playa")
- La respuesta debe ser unicamente una entidad (persona o ciudad o fecha, entre otras), no una descripción o frase larga.

Responde ÚNICAMENTE en formato JSON válido, sin texto adicional:
{{
  "judge_match": 1,
  "judge_justification": "breve explicación de tu decisión"
}}

Donde judge_match debe ser:
- 1 si las respuestas son equivalentes o suficientemente similares
- 0 si son diferentes o incorrectas

JSON:"""

    return prompt


def parse_judge_response(response_text):
    try:
        # Intentar encontrar JSON en la respuesta
        response_text = response_text.strip()

        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        result = json.loads(response_text)

        # Validar que tenga los campos necesarios
        judge_match = int(result.get("judge_match", 0))
        judge_justification = str(result.get("judge_justification", ""))

        return judge_match, judge_justification

    except Exception as e:
        # Si falla el parseo, retornar valores por defecto
        print(f"Error parsing judge response: {e}")
        print(f"Response was: {response_text[:200]}")
        return 0, f"Error en parseo: {str(e)[:100]}"


def process_batch(llm, batch_items, sampling_params):
    prompts = []
    for item in batch_items:
        prompt = create_judge_prompt(
            item['generated_answer'],
            item['respuestas_correctas'],
            item['respuestas_aliases']
        )
        prompts.append(prompt)

    # Generar respuestas del judge
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        judge_response = output.outputs[0].text.strip()
        judge_match, judge_justification = parse_judge_response(judge_response)

        # Crear nuevo item con los campos del judge
        result = batch_items[i].copy()
        result['judge_match'] = judge_match
        result['judge_justification'] = judge_justification

        results.append(result)

    return results


def main(args):
    print(f"Cargando resultados de: {args.input_path}")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total de registros: {len(data)}")

    items_to_judge = data
    print(f"Evaluando todos los items: {len(items_to_judge)}")

    # Temperatura baja para respuestas más determinísticas
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Inicializar modelo judge
    print(f"Cargando modelo judge: {args.judge_model}")
    llm = LLM(
        model=args.judge_model,
        download_dir=args.download_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Procesar en batches
    all_results = []
    curr_idx = 0

    with tqdm(total=len(items_to_judge), desc="Evaluando con Judge") as pbar:
        while curr_idx < len(items_to_judge):
            end_idx = min(curr_idx + args.batch_size, len(items_to_judge))
            batch = items_to_judge[curr_idx:end_idx]

            batch_results = process_batch(llm, batch, sampling_params)
            all_results.extend(batch_results)

            if (curr_idx // args.batch_size) % args.save_every == 0 or end_idx >= len(items_to_judge):

                save_data = all_results

                output_path = Path(args.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)

                print(f"\nProgreso guardado en: {output_path}")

            pbar.update(len(batch))
            curr_idx = end_idx

   
    final_data = all_results

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    total = len(final_data)
    judge_matches = sum(item.get('judge_match', 0) for item in final_data)
    exact_matches = sum(item.get('exact_match', 0) for item in final_data)
    partial_matches = sum(item.get('partial_match', 0) for item in final_data)

    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES:")
    print(f"{'='*60}")
    print(f"Total de ejemplos: {total}")
    print(f"Exact Match: {exact_matches} ({100*exact_matches/total:.2f}%)")
    print(f"Partial Match: {partial_matches} ({100*partial_matches/total:.2f}%)")
    print(f"Judge Match: {judge_matches} ({100*judge_matches/total:.2f}%)")
    print(f"\nMejora con Judge: +{judge_matches - partial_matches} matches adicionales")
    print(f"Resultados guardados en: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge para evaluar similitud semántica en benchmark QA")

    parser.add_argument("--input_path", type=str,default="benchmark_results.json",help="Path al archivo de resultados original")
    parser.add_argument("--output_path", type=str,default="benchmark_results_judged.json",help="Path para guardar resultados con judge")

    parser.add_argument("--judge_model", type=str,default="meta-llama/Llama-3.3-70B-Instruct",help="Modelo a usar como judge")

    parser.add_argument("--download_dir", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument("--temperature", type=float, default=0.1,help="Temperatura para el judge (baja para determinismo)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048,help="Máximo de tokens para respuesta del judge")

    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1000,help="Guardar progreso cada N batches")

    args = parser.parse_args()
    main(args)
