import torch
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from google.colab import drive
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
from huggingface_hub import notebook_login
from io import BytesIO

notebook_login()

# Configuração inicial do Google Drive
try:
    drive.mount('/content/drive', force_remount=False)
except:
    print("Drive já montado")

## ----------------------------
## Classe ModelManager
## ----------------------------
class ModelManager:
    def __init__(self, config_path: str = '/content/drive/MyDrive/biofy-hack/models_ranking.json'):
        self.config_path = config_path
        self.models_data = self._load_models_config()
        self.style_embeddings = self._create_style_embeddings()

    def _load_models_config(self) -> Dict:
        """Carrega configuração dos modelos com tratamento robusto de erros"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                if not isinstance(config.get('models'), dict):
                    raise ValueError("Estrutura inválida do arquivo de configuração")
                return config
        except Exception as e:
            print(f"⚠️ Erro ao carregar configuração: {e}. Usando fallback.")
            return self._get_fallback_config()

    def _get_fallback_config(self) -> Dict:
        """Configuração padrão para fallback"""
        return {
            "models": {
                "SDXL": {
                    "repo": "stabilityai/stable-diffusion-xl-base-1.0",
                    "style": ["realistic", "detailed"],
                    "strengths": ["high quality", "versatility"],
                    "weaknesses": ["speed"],
                    "tags": ["general", "photography"]
                }
            }
        }

    def _create_style_embeddings(self) -> Dict:
        """Cria embeddings para busca semântica de estilos"""
        styles = set()
        for model in self.models_data['models'].values():
            styles.update(model.get('style', []))

        # Embeddings simples (em produção usar modelo de NLP)
        return {style: np.random.rand(50) for style in styles}

    def get_model(self, model_name: str) -> Optional[Dict]:
        return self.models_data['models'].get(model_name)

    def list_models(self) -> List[str]:
        return list(self.models_data['models'].keys())

    def find_similar_styles(self, target_style: str, top_n: int = 3) -> List[str]:
        """Encontra modelos com estilos similares usando embedding"""
        target_embed = self.style_embeddings.get(target_style.lower())
        if not target_embed:
            return []

        scores = []
        for model_name, model_data in self.models_data['models'].items():
            for style in model_data.get('style', []):
                style_embed = self.style_embeddings[style]
                similarity = 1 - cosine(target_embed, style_embed)
                scores.append((model_name, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [model for model, _ in scores[:top_n]]

## ----------------------------
## Classe ModelLoader
## ----------------------------
class ModelLoader:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name: str) -> Optional[Union[StableDiffusionPipeline, StableDiffusionXLPipeline]]:
        """Carrega modelo com otimizações e fallback"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_config = self.model_manager.get_model(model_name)
        if not model_config:
            print(f"Modelo {model_name} não encontrado")
            return None

        try:
            t0 = datetime.now()
            print(f"⏳ Carregando {model_name}...")

            # Configurações comuns adaptáveis à GPU
            common_args = {
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'variant': 'fp16' if torch.cuda.is_available() else None,
                'use_safetensors': True,
                'safety_checker': None
            }


            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                      model_config['repo'],
                      torch_dtype=torch.float16,
                      variant="fp16",
                      use_safetensors=True
                  )
            except Exception as e:
                pipe = StableDiffusionPipeline.from_pretrained(
                      model_config['repo'],
                      torch_dtype=torch.float32,
                      use_safetensors=False
                  )






            # Otimizações com fallback
            pipe = self._optimize_pipeline(pipe)
            pipe = pipe.to(self.device)

            self.loaded_models[model_name] = pipe
            print(f"✅ {model_name} carregado em {(datetime.now()-t0).total_seconds():.1f}s")
            return pipe

        except Exception as e:
            print(f"❌ Falha ao carregar {model_name}: {str(e)}")
            return self._load_fallback_model()

    def _optimize_pipeline(self, pipe):
        """Aplica otimizações de performance com tratamento de erros"""
        try:
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                pipe.enable_xformers_memory_efficient_attention()
        except:
            print("⚠️ Xformers não disponível. Continuando sem otimização.")

        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        return pipe

    def _load_fallback_model(self):
        """Modelo de fallback para recuperação de erros"""
        try:
            print("🔄 Tentando carregar modelo fallback...")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1.5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            return self._optimize_pipeline(pipe).to(self.device)
        except Exception as e:
            print(f"❌ Falha crítica no fallback: {str(e)}")
            return None

    def unload_model(self, model_name: str):
        """Libera memória do modelo"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()
            print(f"♻️ {model_name} descarregado")





## ----------------------------
## Classe ImageGenerator
## ----------------------------
class ImageGenerator:
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.generation_stats = defaultdict(list)

    def generate_image(self, model_name: str, prompt: str, steps: int = 30) -> Tuple[Optional[Image.Image], Dict]:
        """Gera imagem com métricas de performance"""
        pipe = self.model_loader.load_model(model_name)
        if not pipe:
            return None, {}

        try:
            t0 = datetime.now()
            result = pipe(prompt, num_inference_steps=steps)
            gen_time = (datetime.now() - t0).total_seconds()

            metrics = {
                'generation_time': gen_time,
                'memory_usage': torch.cuda.max_memory_allocated() / (1024 ** 2),
                'success': True
            }

            self.generation_stats[model_name].append(metrics)
            return result.images[0], metrics

        except Exception as e:
            print(f"❌ Erro na geração: {str(e)}")
            return None, {'success': False, 'error': str(e)}




## ----------------------------
## Classe FeedbackSystem
## ----------------------------
class FeedbackSystem:
    def __init__(self, feedback_file: str = '/content/drive/MyDrive/biofy-hack/ai_feedback.json'):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()

    def _ensure_global_stats(self):
        """Garante que global_stats tenha a estrutura mínima necessária"""
        if 'global_stats' not in self.feedback_data:
            self.feedback_data['global_stats'] = {
                'repo': 'stabilityai/stable-diffusion-xl-base-1.0',
                'last_updated': str(datetime.now())
            }
        elif 'repo' not in self.feedback_data['global_stats']:
            self.feedback_data['global_stats']['repo'] = 'stabilityai/stable-diffusion-xl-base-1.0'

    def _load_feedback(self) -> Dict:
        """Carrega feedback existente com tratamento de erros"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    # Garante que a estrutura tenha todos os campos necessários
                    if 'interactions' not in data:
                        data['interactions'] = []
                    if 'model_stats' not in data:
                        data['model_stats'] = {}

                    self._ensure_global_stats()
                    return data
        except Exception as e:
            print(f"⚠️ Erro ao carregar feedback: {str(e)}")

        # Retorna estrutura padrão se o arquivo não existir ou estiver corrompido
        return {
            'interactions': [],
            'model_stats': {},
            'global_stats': {
                'repo': 'stabilityai/stable-diffusion-xl-base-1.0',
                'last_updated': str(datetime.now())
            }
        }

    def save_feedback(self, user_id: str, model_name: str, prompt: str, rating: int, gen_metrics: Dict):
        """Registra feedback com métricas completas"""
        feedback_entry = {
            'user_id': user_id,
            'model': model_name,
            'prompt': prompt,
            'rating': rating,
            'timestamp': str(datetime.now()),
            'metrics': gen_metrics
        }

        self.feedback_data['interactions'].append(feedback_entry)

        # Inicializa estatísticas do modelo se não existirem
        if model_name not in self.feedback_data['model_stats']:
            self.feedback_data['model_stats'][model_name] = {
                'total_uses': 0,
                'avg_rating': 0,
                'total_ratings': 0
            }

        stats = self.feedback_data['model_stats'][model_name]
        stats['total_uses'] += 1

        if rating > 0:
            stats['total_ratings'] += 1
            stats['avg_rating'] = ((stats['avg_rating'] * (stats['total_ratings'] - 1)) + rating) / stats['total_ratings']

        self._save_to_disk()

    def _save_to_disk(self):
        """Salva dados no Google Drive"""
        try:
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            print(f"❌ Falha ao salvar feedback: {str(e)}")

    def get_model_stats(self, model_name: str) -> Dict:
        """Obtém estatísticas de uso do modelo"""
        return self.feedback_data['model_stats'].get(model_name, {
            'total_uses': 0,
            'avg_rating': 0,
            'total_ratings': 0
        })

    def get_global_stats(self) -> Dict:
        """Obtém estatísticas globais com fallback seguro"""
        return self.feedback_data.get('global_stats', {
            'repo': 'stabilityai/stable-diffusion-xl-base-1.0',
            'last_updated': str(datetime.now())
        })



## ----------------------------
## Classe AILearningSystem (Principal)
## ----------------------------
class AILearningSystem:
    def __init__(self):
        self.model_manager = ModelManager()
        self.model_loader = ModelLoader(self.model_manager)
        self.image_generator = ImageGenerator(self.model_loader)
        self.feedback_system = FeedbackSystem()

    def analyze_prompt(self, prompt: str) -> Dict:
        """Analisa o prompt para recomendação de modelos"""
        prompt_lower = prompt.lower()
        features = {
            'styles': [],
            'requirements': [],
            'tags': []
        }

        # Detecção de estilo
        for style in self.model_manager.style_embeddings.keys():
            if style in prompt_lower:
                features['styles'].append(style)

        # Requisitos técnicos
        if any(kw in prompt_lower for kw in ['detalhado', 'detalhes', 'complexo']):
            features['requirements'].append('high_detail')
        if any(kw in prompt_lower for kw in ['rápido', 'veloz', 'speed']):
            features['requirements'].append('speed')

        return features

    def recommend_models(self, prompt: str, top_n: int = 3) -> List[str]:
        """Recomenda modelos usando múltiplos critérios"""
        features = self.analyze_prompt(prompt)
        scored_models = []

        for model_name in self.model_manager.list_models():
            score = 0
            model_data = self.model_manager.get_model(model_name)
            model_stats = self.feedback_system.get_model_stats(model_name)

            # 1. Pontua por correspondência de estilo (50%)
            style_match = sum(1 for style in features['styles'] if style in model_data.get('style', []))
            score += style_match * 0.5

            # 2. Pontua por avaliações anteriores (30%)
            score += model_stats['avg_rating'] * 0.3

            # 3. Pontua por requisitos especiais (20%)
            if 'high_detail' in features['requirements'] and 'high quality' in model_data.get('strengths', []):
                score += 0.2
            if 'speed' in features['requirements'] and 'fast' in model_data.get('strengths', []):
                score += 0.1

            scored_models.append((model_name, score))

        # Ordena e seleciona os melhores
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return [model for model, score in scored_models[:top_n]]

    def generate_comparison(self, prompt: str, num_models: int = 3) -> Dict:
        """Gera imagens comparativas"""
        recommended = self.recommend_models(prompt, num_models)
        results = []

        for model in recommended:
            image, metrics = self.image_generator.generate_image(model, prompt)
            results.append({
                'model': model,
                'image': image,
                'metrics': metrics,
                'stats': self.feedback_system.get_model_stats(model)
            })

        return {
            'prompt': prompt,
            'results': results,
            'timestamp': str(datetime.now())
        }







    def api_generate(self, user_id: str, prompt: str) -> Dict:
        """Endpoint para geração via API"""
        comparison = self.generate_comparison(prompt)

        # Formata resposta para API
        response = {
            "request_id": str(hash(f"{user_id}{prompt}")),  # ID único para referência
            "prompt": prompt,
            "results": []
        }

        for result in comparison['results']:
            if result['image']:
                # Converte imagem para base64
                buffered = BytesIO()
                result['image'].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                response['results'].append({
                    "model": result['model'],
                    "image_base64": img_str,
                    "metrics": result['metrics'],
                    "stats": result['stats']
                })

        return response

    def api_submit_feedback(self, request_id: str, model_name: str, rating: int) -> Dict:
        """Endpoint para envio de feedback"""
        # Nota: Em produção, você armazenaria os requests temporariamente
        self.feedback_system.save_feedback(
            user_id="API_USER_" + request_id[:8],
            model_name=model_name,
            prompt="From API request",  # Você pode armazenar o prompt real se necessário
            rating=rating,
            gen_metrics={}  # Pode adicionar métricas se disponíveis
        )
        return {"status": "success", "message": "Feedback registered"}

