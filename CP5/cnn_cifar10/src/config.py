"""
config.py
---------
Ponto único de configuração do projeto.

Por que centralizar? Porque um experimento de aprendizado de máquina só é
reprodutível se todos os "botões" (hiperparâmetros, caminhos, semente
aleatória) estiverem escritos em um lugar só, versionados junto com o código.
Espalhar números mágicos pelos scripts é a causa nº 1 de resultados que
"funcionavam ontem" e não funcionam hoje.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos: sempre relativos à raiz do projeto, nunca ao diretório de onde
# você chamou o script. parents[1] sobe de src/ para cnn/.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"        # onde o dataset bruto é baixado/armazenado
OUT_DIR = ROOT / "outputs"      # onde salvamos pesos, gráficos e métricas


@dataclass
class Config:
    # --- Dados ---------------------------------------------------------
    dataset: str = "CIFAR10"
    tamanho_imagem: int = 32          # imagens 32x32
    canais: int = 3                   # 1 = escala de cinza, 3 = RGB
    n_classes: int = 10
    frac_validacao: float = 0.1       # 10% do treino vira validação
    num_workers: int = 0              # 0 é o mais seguro no Windows

    # Estatísticas do CIFAR-10 (calculadas por canal sobre o conjunto de
    # treino). Diferente do FashionMNIST (1 canal), aqui cada tupla tem 3
    # valores, um por canal RGB. Usadas para normalizar: x = (x - media) / desvio
    media: tuple = (0.4914, 0.4822, 0.4465)
    desvio: tuple = (0.2470, 0.2435, 0.2616)

    # --- Treinamento ---------------------------------------------------
    batch_size: int = 128
    epocas: int = 20                  # CIFAR-10 é mais difícil: mais épocas
    lr: float = 1e-3                  # taxa de aprendizado do Adam
    weight_decay: float = 1e-4        # regularização L2
    paciencia: int = 5                # early stopping (curva mais ruidosa, mais paciência)
    semente: int = 42

    # --- Arquitetura ---------------------------------------------------
    # Um bloco extra e mais filtros por bloco em relação ao FashionMNIST
    # ((16,32,64) -> (32,64,128,256)): fotos coloridas de objetos/animais em
    # cenas reais (fundo variado, iluminação, pose, oclusão) têm muito mais
    # variação visual do que roupas centralizadas em fundo uniforme, então a
    # rede precisa de mais filtros e mais profundidade para separar as classes.
    canais_conv: tuple = (32, 64, 128, 256)  # filtros de cada bloco convolucional
    neuronios_fc: int = 256
    dropout: float = 0.4              # mais overfitting nesse dataset -> mais dropout

    # --- Arquivos de saída ---------------------------------------------
    arq_checkpoint: str = "melhor_modelo.pt"
    arq_historico: str = "historico.json"
    arq_curvas: str = "curvas_treino.png"
    arq_confusao: str = "matriz_confusao.png"
    arq_metricas: str = "metricas_teste.json"
    arq_torchscript: str = "modelo_scriptado.pt"
    arq_classes: str = "classes.json"

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()

# Nomes legíveis das 10 classes do CIFAR-10, na ordem dos rótulos 0..9.
CLASSES = [
    "Avião",
    "Automóvel",
    "Pássaro",
    "Gato",
    "Cervo",
    "Cachorro",
    "Sapo",
    "Cavalo",
    "Navio",
    "Caminhão",
]
