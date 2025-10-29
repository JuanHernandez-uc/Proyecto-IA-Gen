import argparse
import json
import torch
import ast
from pathlib import Path
from datasets import Dataset
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer
import unicodedata
import re
import os

login(token="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")

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


#formatear texto
def remove_accents(text):
    if not isinstance(text, str):
        return str(text)
    text = text.lower()
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
def parse_list_field(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []
    if pd.isna(value):
        return []
    return [value]
def normalize_answer(answer):
    if not isinstance(answer, str):
        return ""
    answer = remove_accents(answer)
    answer = answer.strip(".,;:¿?¡!\"'()[]")
    return re.sub(r'\s+', ' ', answer).strip()

def load_qa_dataset(csv_path, sample_size=None):
    df = pd.read_csv(csv_path)
    
    df['respuestas'] = df['respuestas'].apply(parse_list_field)
    df['respuestas_aliases'] = df['respuestas_aliases'].apply(parse_list_field)
    df['objetos'] = df['objetos'].apply(parse_list_field)
    df['obtenido_de'] = df['obtenido_de'].apply(parse_list_field)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    return df


def prepare_qa_prompt(row, tokenizer, system_prompt=None):
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row['pregunta']}
    ]
    
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def extract_all_valid_answers(row):
    valid_answers = set()
    
    respuestas = row['respuestas']
    if isinstance(respuestas, list):
        for r in respuestas:
            if isinstance(r, str) and r.strip():
                valid_answers.add(normalize_answer(r))
    
    respuestas_aliases = row['respuestas_aliases']
    if isinstance(respuestas_aliases, list):
        for alias_group in respuestas_aliases:
            if isinstance(alias_group, list):
                for alias in alias_group:
                    if isinstance(alias, str) and alias.strip():
                        valid_answers.add(normalize_answer(alias))
            elif isinstance(alias_group, str) and alias_group.strip():
                valid_answers.add(normalize_answer(alias_group))
    
    return valid_answers

####### REVISARRRRRRRRR
def evaluate_answer(generated, valid_answers):
    generated_norm = normalize_answer(generated)
    
    if generated_norm in valid_answers:
        return 1, 1
    
    for valid_ans in valid_answers:
        if valid_ans and valid_ans in generated_norm:
            return 0, 1
    
    for valid_ans in valid_answers:
        if valid_ans and valid_ans in generated_norm.split(','):
            generated_parts = [normalize_answer(p) for p in generated.split(',')]
            if valid_ans in generated_parts:
                return 1, 1
    
    return 0, 0


def process_batch(llm, batch, sampling_params):
    prompts = list(batch["eval_prompt"])
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        valid_answers = batch['valid_answers'][i]
        exact_match, partial_match = evaluate_answer(generated_text, valid_answers)
        
        results.append({
            'idx': batch['idx'][i],
            'entidad': batch['entidad'][i],
            'relacion': batch['relacion'][i],
            'pregunta': batch['pregunta'][i],
            'respuestas_correctas': batch['respuestas'][i],
            'respuestas_aliases': batch['respuestas_aliases'][i],
            'generated_answer': generated_text,
            'exact_match': exact_match,
            'partial_match': partial_match,
            'pais': batch['obtenido_de'][i] if 'obtenido_de' in batch else None
        })
    
    return results


def main(args):

    # Cargar y preparar dataset
    df = load_qa_dataset(args.qa_dataset_path, sample_size=args.sample_size)
    df['idx'] = range(len(df))
    df['valid_answers'] = df.apply(extract_all_valid_answers, axis=1)
    dataset = Dataset.from_pandas(df)
    
    # Cargar tokenizer y preparar prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    with open(args.system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    dataset = dataset.map(
        lambda x: {"eval_prompt": prepare_qa_prompt(x, tokenizer, system_prompt)},
        desc="Preparando prompts"
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ####Qwens pequeños ponerles el Stop 
        stop=args.stop_sequences if args.stop_sequences else None
    )
    
    llm = LLM(
        model=args.model_path,
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
    
    with tqdm(total=len(dataset), desc="Procesando QA") as pbar:
        while curr_idx < len(dataset):
            end_idx = min(curr_idx + args.batch_size, len(dataset))
            curr_batch = dataset[curr_idx:end_idx]
            
            batch_results = process_batch(llm, curr_batch, sampling_params)
            all_results.extend(batch_results)
            
            if (curr_idx // args.batch_size) % args.save_every == 0 or end_idx >= len(dataset):
                output_path = Path(args.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            curr_idx = end_idx
            pbar.update(end_idx - curr_idx + args.batch_size)
    
    # Calcular métricas finales
    total = len(all_results)
    exact_matches = sum(r['exact_match'] for r in all_results)
    partial_matches = sum(r['partial_match'] for r in all_results)
    
    relations_stats = {}
    for r in all_results:
        rel = r['relacion']
        if rel not in relations_stats:
            relations_stats[rel] = {'total': 0, 'exact': 0, 'partial': 0}
        relations_stats[rel]['total'] += 1
        relations_stats[rel]['exact'] += r['exact_match']
        relations_stats[rel]['partial'] += r['partial_match']
    
    metrics = {
        'total_examples': total,
        'exact_match': exact_matches,
        'exact_match_pct': 100 * exact_matches / total,
        'partial_match': partial_matches,
        'partial_match_pct': 100 * partial_matches / total,
        'by_relation': {
            rel: {
                'total': stats['total'],
                'exact_match': stats['exact'],
                'exact_match_pct': 100 * stats['exact'] / stats['total'],
                'partial_match': stats['partial'],
                'partial_match_pct': 100 * stats['partial'] / stats['total'],
            }
            for rel, stats in relations_stats.items()
        }
    }
    
    metrics_path = Path(args.output_path).parent / f"{Path(args.output_path).stem}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\nResultados: Exact Match: {100*exact_matches/total:.2f}% | Partial Match: {100*partial_matches/total:.2f}%")
    print(f"Guardados en: {args.output_path} y {metrics_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark de QA sobre modelos 7B con vLLM")
    
    parser.add_argument("--qa_dataset_path", type=str, default="QA/QA_dataset_aliases_filtrados.csv")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-7B-Instruct")

    parser.add_argument("--download_dir", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--stop_sequences", type=str, nargs="*", default=None)

    parser.add_argument("--output_path", type=str, default="QA/benchmark_results.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=5)
    
    args = parser.parse_args()
    main(args)