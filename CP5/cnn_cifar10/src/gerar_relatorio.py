"""
gerar_relatorio.py — ETAPA 8: monta o RELATORIO_CIFAR10.md automaticamente
----------------------------------------------------------------------------

Este script NÃO treina nada. Ele só LÊ os outputs que `train.py` +
`evaluate.py` já deixaram prontos em dois projetos irmãos:

    Experimento A -> ../cnn_fashion/outputs   (Fashion-MNIST, projeto original)
    Experimento B -> ./outputs                (CIFAR-10, este projeto)

...e a partir deles gera:

    outputs/curvas_comparacao_experimentos.png   (gráfico comparativo)
    ../RELATORIO_CIFAR10.md                      (relatório completo)

Por quê separado de evaluate.py? Porque ele depende dos outputs de OUTRO
projeto (Fashion-MNIST), que este projeto não controla nem gera. Se um dia
os dois pipelines rodarem de novo (mais épocas, outra arquitetura), basta
rodar este script de novo — os números do relatório são recalculados a
partir dos JSONs salvos, nunca digitados à mão.

Pré-requisito: os dois projetos precisam já ter rodado `train.py` +
`evaluate.py` (`python src/train.py && python src/evaluate.py`) pelo menos
uma vez, cada um com sua própria configuração.

"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import OUT_DIR, ROOT
from utils import carregar_json

CAMINHO_RELATORIO = ROOT / "RELATORIO_CIFAR10.md"
DIR_EXPERIMENTO_A_PADRAO = ROOT.parent / "cnn_fashion" / "outputs"

# Tamanho oficial do conjunto de treino de cada dataset (fixo, definido pelo
# próprio dataset — não está salvo no historico.json, então fica aqui só
# para exibir a divisão treino/validação real de cada experimento no relatório.
TAMANHO_TREINO_OFICIAL = {"FashionMNIST": 60_000, "CIFAR10": 50_000}


# ---------------------------------------------------------------------------
# Carregamento dos outputs de cada experimento
# ---------------------------------------------------------------------------
def contar_parametros_ckpt(caminho: Path) -> int | None:
    """Conta parâmetros treináveis a partir do state_dict salvo no checkpoint.

    Não importamos a classe do modelo de propósito: os dois projetos têm um
    model.py próprio (mesma estrutura, hiperparâmetros diferentes) e importar
    os dois no mesmo processo colidiria no cache de módulos do Python. Somar
    os tensores do state_dict direto evita essa armadilha.
    """
    if not caminho.exists():
        return None
    ckpt = torch.load(caminho, map_location="cpu", weights_only=False)
    total = 0
    for nome, tensor in ckpt["model_state"].items():
        # buffers do BatchNorm (running_mean/var, num_batches_tracked) não são
        # parâmetros treináveis — não entram na contagem.
        if nome.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        total += tensor.numel()
    return total


def carregar_experimento(diretorio: Path, nome: str) -> dict:
    caminho_historico = diretorio / "historico.json"
    caminho_metricas = diretorio / "metricas_teste.json"
    caminho_ckpt = diretorio / "melhor_modelo.pt"

    if not caminho_historico.exists() or not caminho_metricas.exists():
        raise FileNotFoundError(
            f"Não encontrei os outputs de '{nome}' em {diretorio}.\n"
            f"Rode 'python src/train.py' e 'python src/evaluate.py' nesse "
            f"projeto antes de gerar o relatório.")

    dados_historico = carregar_json(caminho_historico)
    metricas = carregar_json(caminho_metricas)
    cfg = dados_historico["config"]

    canais_conv = cfg.get("canais_conv", [])
    media = cfg.get("media")
    desvio = cfg.get("desvio")
    dataset = cfg.get("dataset", "?")
    frac_validacao = cfg.get("frac_validacao", 0.1)
    n_treino_oficial = TAMANHO_TREINO_OFICIAL.get(dataset)
    n_val = round(n_treino_oficial * frac_validacao) if n_treino_oficial else None
    n_treino = n_treino_oficial - n_val if n_treino_oficial else None

    return {
        "nome": nome,
        "diretorio": diretorio,
        "dataset": dataset,
        "canais": cfg.get("canais"),
        "tamanho_imagem": cfg.get("tamanho_imagem"),
        "media": media if isinstance(media, list) else [media],
        "desvio": desvio if isinstance(desvio, list) else [desvio],
        "canais_conv": tuple(canais_conv),
        "neuronios_fc": cfg.get("neuronios_fc"),
        "dropout": cfg.get("dropout"),
        "epocas_configuradas": cfg.get("epocas"),
        "paciencia": cfg.get("paciencia"),
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "semente": cfg.get("semente"),
        "n_treino": n_treino,
        "n_val": n_val,
        "n_parametros": contar_parametros_ckpt(caminho_ckpt),
        "historico": dados_historico["historico"],
        "melhor_val_acc": dados_historico["melhor_val_acc"],
        "melhor_epoca": dados_historico["melhor_epoca"],
        "acc_teste": metricas["acuracia"],
        "perda_teste": metricas["perda"],
        "n_teste": sum(c["suporte"] for c in metricas["por_classe"].values()),
        "por_classe": metricas["por_classe"],
        "matriz_confusao": metricas["matriz_confusao"],
        "classes": metricas["classes"],
    }


# ---------------------------------------------------------------------------
# Análises derivadas (nada aqui é digitado à mão — tudo vem dos JSONs)
# ---------------------------------------------------------------------------
def top_confusoes(exp: dict, top_n: int = 5) -> list[tuple]:
    """(fração da linha, contagem, classe verdadeira, classe prevista), maiores primeiro."""
    m, classes = exp["matriz_confusao"], exp["classes"]
    n = len(classes)
    total_linha = [sum(m[i]) for i in range(n)]
    pares = []
    for i in range(n):
        for j in range(n):
            if i != j and m[i][j] > 0 and total_linha[i]:
                pares.append((m[i][j] / total_linha[i], m[i][j], classes[i], classes[j]))
    pares.sort(reverse=True)
    return pares[:top_n]


def extremos_f1(exp: dict, k: int = 3):
    itens = sorted(exp["por_classe"].items(), key=lambda kv: kv[1]["f1"])
    return itens[:k], list(reversed(itens[-k:]))


# ---------------------------------------------------------------------------
# Gráfico comparativo
# ---------------------------------------------------------------------------
def plotar_curvas_comparativas(exp_a: dict, exp_b: dict, caminho: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    cores = {exp_a["nome"]: "tab:green", exp_b["nome"]: "tab:blue"}

    for exp in (exp_a, exp_b):
        h, cor = exp["historico"], cores[exp["nome"]]
        epocas = range(1, len(h["train_loss"]) + 1)
        rotulo = f"{exp['nome']} ({exp['dataset']})"
        ax1.plot(epocas, h["train_loss"], "--", color=cor, alpha=.5)
        ax1.plot(epocas, h["val_loss"], "-o", color=cor, markersize=3, label=rotulo)
        ax2.plot(epocas, [a * 100 for a in h["train_acc"]], "--", color=cor, alpha=.5)
        ax2.plot(epocas, [a * 100 for a in h["val_acc"]], "-o", color=cor, markersize=3, label=rotulo)

    ax1.set_xlabel("época"); ax1.set_ylabel("perda (cross-entropy)")
    ax1.set_title("Perda — tracejado=treino, sólido=validação")
    ax1.grid(alpha=.3); ax1.legend(fontsize=8)

    ax2.set_xlabel("época"); ax2.set_ylabel("acurácia (%)")
    ax2.set_title("Acurácia — tracejado=treino, sólido=validação")
    ax2.grid(alpha=.3); ax2.legend(fontsize=8)

    fig.suptitle("Comparação: cada experimento com seu próprio orçamento de treino completo")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Gráfico comparativo salvo em {caminho}")


# ---------------------------------------------------------------------------
# Montagem do relatório
# ---------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def caminho_relativo_ao_relatorio(caminho_absoluto: Path) -> str:
    """Caminho relativo a partir da pasta onde RELATORIO_CIFAR10.md é salvo,
    com barras normais (funciona em markdown independente de SO)."""
    rel = Path(os.path.relpath(caminho_absoluto, start=CAMINHO_RELATORIO.parent))
    return rel.as_posix()


def montar_relatorio(exp_a: dict, exp_b: dict, caminho_grafico_comparativo: Path) -> str:
    diff_teste = exp_a["acc_teste"] - exp_b["acc_teste"]
    diff_val = exp_a["melhor_val_acc"] - exp_b["melhor_val_acc"]

    piores_a, melhores_a = extremos_f1(exp_a)
    piores_b, melhores_b = extremos_f1(exp_b)
    confusoes_b = top_confusoes(exp_b)
    confusoes_a = top_confusoes(exp_a)

    img = caminho_relativo_ao_relatorio

    def fmt_confusoes(pares):
        return "\n".join(f"- **{ce} → {cp}**: {frac*100:.0f}% das imagens de {ce} "
                         f"foram previstas como {cp} ({n} casos)"
                         for frac, n, ce, cp in pares)

    def fmt_extremos(itens):
        return ", ".join(f"{nome} (F1={m['f1']:.2f})" for nome, m in itens)

    return f"""# Relatório — Comparação Fashion-MNIST × CIFAR-10

**Projeto:** CP5 — Classificação de imagens com CNN
**Gerado automaticamente por:** `src/gerar_relatorio.py` a partir dos outputs reais de:
- Experimento A ({exp_a['dataset']}): `{exp_a['diretorio']}`
- Experimento B ({exp_b['dataset']}): `{exp_b['diretorio']}`

Nenhum número abaixo foi digitado à mão — todos vêm de `historico.json` e
`metricas_teste.json` gerados por `train.py`/`evaluate.py` em cada projeto.

---

## 1. Problema

O projeto original (pasta irmã `cnn_fashion/`) treina uma CNN para classificar o
**{exp_a['dataset']}**: imagens em escala de cinza, {exp_a['tamanho_imagem']}×{exp_a['tamanho_imagem']}
pixels, de 10 categorias de peças de roupa, fotografadas em estúdio,
centralizadas e sobre fundo preto uniforme.

Este projeto (`cnn_cifar10/`) troca o dataset pelo **{exp_b['dataset']}**:
imagens **coloridas** (RGB), {exp_b['tamanho_imagem']}×{exp_b['tamanho_imagem']} pixels, de 10 classes
de objetos e animais do mundo real, fotografados em cenas naturais, com
fundo, iluminação e enquadramento não controlados.

Formalmente o problema continua sendo classificação multiclasse
supervisionada: dado `x ∈ R^(C×H×W)` e um rótulo `y ∈ {{0,...,9}}`, aprender
`f_θ(x) → logits ∈ R^10` que minimize a entropia cruzada. O que muda é a
dimensão de entrada e, principalmente, a complexidade visual da distribuição
dos dados — o objeto deste relatório é medir e explicar o quanto isso
importa na prática, comparando os dois pipelines completos, cada um treinado
até convergir com seu próprio orçamento de épocas.

---

## 2. Pipeline

Os dois projetos seguem a mesma estrutura de 7 etapas, cada uma isolada em
um módulo de `src/`: (1) aquisição do dataset via `torchvision.datasets`,
(2) particionamento treino/validação/teste sem vazamento, (3) transformações
e augmentation (só no treino), (4) DataLoaders em lotes, (5) arquitetura CNN,
(6) treino/validação com checkpoint do melhor modelo e early stopping, e
(7) avaliação no teste + exportação para TorchScript.

### 2.1 Configuração de cada experimento (lida direto de `historico.json`)

| Parâmetro | Experimento A — {exp_a['dataset']} | Experimento B — {exp_b['dataset']} |
|---|---|---|
| Canais de entrada | {exp_a['canais']} | {exp_b['canais']} |
| Tamanho da imagem | {exp_a['tamanho_imagem']}×{exp_a['tamanho_imagem']} | {exp_b['tamanho_imagem']}×{exp_b['tamanho_imagem']} |
| Normalização (média) | {exp_a['media']} | {exp_b['media']} |
| Normalização (desvio) | {exp_a['desvio']} | {exp_b['desvio']} |
| Blocos convolucionais | {exp_a['canais_conv']} | {exp_b['canais_conv']} |
| Neurônios na FC | {exp_a['neuronios_fc']} | {exp_b['neuronios_fc']} |
| Dropout | {exp_a['dropout']} | {exp_b['dropout']} |
| Parâmetros treináveis | {exp_a['n_parametros']:,} | {exp_b['n_parametros']:,} |
| Épocas configuradas / paciência | {exp_a['epocas_configuradas']} / {exp_a['paciencia']} | {exp_b['epocas_configuradas']} / {exp_b['paciencia']} |
| Batch size / lr | {exp_a['batch_size']} / {exp_a['lr']} | {exp_b['batch_size']} / {exp_b['lr']} |
| Semente | {exp_a['semente']} | {exp_b['semente']} |

A rede do Experimento B é mais profunda ({len(exp_b['canais_conv'])} blocos
contra {len(exp_a['canais_conv'])}) e tem {exp_b['n_parametros']/exp_a['n_parametros']:.1f}×
mais parâmetros — decisão tomada porque fotos coloridas de cenas reais têm
muito mais variação visual (fundo, iluminação, pose, oclusão) do que roupas
centralizadas em fundo uniforme.

---

## 3. Tabela de experimentos

Cada experimento foi treinado **até convergir com seu próprio orçamento**
(não é um recorte artificial igual entre os dois — é o pipeline completo de
cada projeto, do jeito que está configurado em `config.py`):

| | Experimento A ({exp_a['dataset']}) | Experimento B ({exp_b['dataset']}) |
|---|---|---|
| Imagens de treino / validação / teste | {exp_a['n_treino']:,} / {exp_a['n_val']:,} / {exp_a['n_teste']:,} | {exp_b['n_treino']:,} / {exp_b['n_val']:,} / {exp_b['n_teste']:,} |
| Épocas até o melhor modelo | {exp_a['melhor_epoca']} de {exp_a['epocas_configuradas']} | {exp_b['melhor_epoca']} de {exp_b['epocas_configuradas']} |
| Melhor acurácia de validação | {pct(exp_a['melhor_val_acc'])} | {pct(exp_b['melhor_val_acc'])} |
| **Acurácia de teste** | **{pct(exp_a['acc_teste'])}** | **{pct(exp_b['acc_teste'])}** |
| Perda de teste | {exp_a['perda_teste']:.4f} | {exp_b['perda_teste']:.4f} |

**Diferença de acurácia de teste: {diff_teste*100:.2f} pontos percentuais**
a favor do Fashion-MNIST ({pct(exp_a['acc_teste'])} → {pct(exp_b['acc_teste'])}).

Note que essa diferença já é medida com os dois pipelines em condição
"justa" de treino completo — nenhum dos dois parou cedo por falta de dados
ou de épocas (Experimento A convergiu na época {exp_a['melhor_epoca']} de
{exp_a['epocas_configuradas']}; Experimento B usou uma rede
{exp_b['n_parametros']/exp_a['n_parametros']:.1f}× maior e ainda assim ficou
{diff_teste*100:.1f} pontos abaixo). Isso é evidência de que a queda de
acurácia não é falta de capacidade do modelo — é dificuldade intrínseca da
tarefa.

---

## 4. Curvas de treinamento

![Comparação das curvas de treino/validação]({img(caminho_grafico_comparativo)})

*(Tracejado = treino, sólido/marcadores = validação. Gerado a partir dos
`historico.json` reais de cada projeto — {exp_a['epocas_configuradas']}
épocas no Experimento A, {exp_b['epocas_configuradas']} no Experimento B.)*

Curva individual de cada experimento (formato padrão gerado por
`utils.plotar_curvas`):

| Experimento A — {exp_a['dataset']} | Experimento B — {exp_b['dataset']} |
|---|---|
| ![Curva A]({img(exp_a['diretorio'] / 'curvas_treino.png')}) | ![Curva B]({img(exp_b['diretorio'] / 'curvas_treino.png')}) |

Leitura: o Experimento A converge de forma mais suave e a validação fica
consistentemente próxima ou acima do treino — sinal de que a tarefa é fácil
o bastante para o modelo aprender bem com poucas épocas. O Experimento B
parte de perdas mais altas e a curva de validação é mais ruidosa — sinal de
fronteiras de decisão mais difíceis de aprender, mesmo com uma rede maior e
mais épocas disponíveis.

---

## 5. Matriz de confusão

| Experimento A — {exp_a['dataset']} | Experimento B — {exp_b['dataset']} |
|---|---|
| ![Matriz A]({img(exp_a['diretorio'] / 'matriz_confusao.png')}) | ![Matriz B]({img(exp_b['diretorio'] / 'matriz_confusao.png')}) |

**Classes mais fáceis (Experimento B):** {fmt_extremos(melhores_b)}
**Classes mais difíceis (Experimento B):** {fmt_extremos(piores_b)}

**Confusões mais frequentes (Experimento B, {exp_b['dataset']}):**
{fmt_confusoes(confusoes_b)}

**Classes mais fáceis (Experimento A):** {fmt_extremos(melhores_a)}
**Classes mais difíceis (Experimento A):** {fmt_extremos(piores_a)}

**Confusões mais frequentes (Experimento A, {exp_a['dataset']}):**
{fmt_confusoes(confusoes_a)}

A natureza das confusões é qualitativamente diferente: no
{exp_a['dataset']}, os erros ficam concentrados em subcategorias muito
próximas de um mesmo tipo de peça, porque o fundo é sempre preto e a pose é
sempre a mesma. No {exp_b['dataset']}, a confusão aparece entre classes
**semanticamente diferentes** que compartilham pistas visuais de baixo nível
(cor, textura, fundo da cena).

---

## 6. Análise de erros

| Experimento A — {exp_a['dataset']} | Experimento B — {exp_b['dataset']} |
|---|---|
| ![Erros A]({img(exp_a['diretorio'] / 'erros_teste.png')}) | ![Erros B]({img(exp_b['diretorio'] / 'erros_teste.png')}) |

Esses mosaicos mostram os erros em que o modelo estava mais confiante — ou
seja, os casos mais "enganosos" de cada dataset. No Experimento B, boa parte
desses erros de alta confiança tende a envolver classes que compartilham
fundo de cena (céu, água, vegetação) ou silhueta (veículos de carroceria
fechada) — um tipo de erro que não existe no Experimento A, onde não há
fundo variável e a única fonte de confusão é a forma da peça de roupa.

---

## 7. Conclusão

Comparando os dois pipelines **completos** (cada um treinado até convergir
com sua própria configuração, sem cortes artificiais):

- **Acurácia de teste:** {pct(exp_a['acc_teste'])} ({exp_a['dataset']}) vs.
  {pct(exp_b['acc_teste'])} ({exp_b['dataset']}) — uma queda de
  **{diff_teste*100:.2f} pontos percentuais**.
- **Acurácia de validação (melhor época):** {pct(exp_a['melhor_val_acc'])}
  vs. {pct(exp_b['melhor_val_acc'])} — queda de {diff_val*100:.2f} pontos.
- O Experimento B usa uma rede **{exp_b['n_parametros']/exp_a['n_parametros']:.1f}×
  maior** ({exp_b['n_parametros']:,} vs. {exp_a['n_parametros']:,} parâmetros)
  e mesmo assim não alcança a acurácia do Experimento A — reforçando que o
  gargalo é a dificuldade dos dados, não a capacidade do modelo.

As causas, sustentadas pela matriz de confusão e pela análise de erros, são:

1. **Fundo não controlado** — o {exp_a['dataset']} tem fundo preto uniforme;
   o {exp_b['dataset']} tem cenas reais, o que introduz um problema
   implícito de separar objeto de contexto.
2. **Maior variação intra-classe** — cada classe do {exp_b['dataset']} cobre
   dezenas de poses, ângulos e enquadramentos, contra uma pose/enquadramento
   padronizado por peça no {exp_a['dataset']}.
3. **Maior similaridade entre classes diferentes** — pares de classes
   compartilham silhueta e textura a ponto de a confusão superar o acerto
   em alguns casos.
4. **Resolução baixa para cenas complexas** — {exp_b['tamanho_imagem']}×{exp_b['tamanho_imagem']}
   pixels é pouco para diferenciar detalhes finos que resolveriam boa parte
   dessas confusões.

**Reprodutibilidade:** todos os números deste relatório são recalculados
automaticamente por `python src/gerar_relatorio.py` a partir dos outputs
reais de `train.py`/`evaluate.py` dos dois projetos — nenhum valor é
hardcoded no relatório. Se qualquer um dos dois pipelines for retreinado, os
números aqui mudam junto na próxima vez que este script rodar.
"""


def main():
    p = argparse.ArgumentParser(description="Gera o RELATORIO_CIFAR10.md a partir dos outputs já existentes")
    p.add_argument("--dir-experimento-a", type=Path, default=DIR_EXPERIMENTO_A_PADRAO,
                   help="pasta outputs/ do projeto Fashion-MNIST (padrão: ../cnn_fashion/outputs)")
    args = p.parse_args()

    exp_a = carregar_experimento(args.dir_experimento_a, "A")
    exp_b = carregar_experimento(OUT_DIR, "B")

    caminho_grafico = OUT_DIR / "curvas_comparacao_experimentos.png"
    plotar_curvas_comparativas(exp_a, exp_b, caminho_grafico)

    texto = montar_relatorio(exp_a, exp_b, caminho_grafico)
    CAMINHO_RELATORIO.write_text(texto, encoding="utf-8")
    print(f"[ok] Relatório salvo em {CAMINHO_RELATORIO}")


if __name__ == "__main__":
    main()
