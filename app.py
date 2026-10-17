"""
Análise de Crédito para Financiamento de Moto
---------------------------------------------
Aplicação interna (Streamlit + TensorFlow/Keras).

Fluxo:
1. O funcionário informa Idade, Renda mensal e Valor da parcela desejada.
2. Uma rede neural (treinada com dados sintéticos baseados em uma política
   de crédito fictícia) calcula a probabilidade de aprovação.
3. O app exibe "Foi aprovado" ou "Não aprovado".
4. Se for negado, sugere uma parcela menor que seria aprovada.

Executar localmente:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÕES E POLÍTICA DE CRÉDITO (regras usadas para gerar os dados)
# ---------------------------------------------------------------------------
IDADE_MIN, IDADE_MAX = 18, 75        # faixa etária aceita
RENDA_MINIMA = 1500.0                # renda mínima mensal (R$)
COMPROMETIMENTO_MAX = 0.30           # parcela pode ser no máx. 30% da renda
LIMIAR_APROVACAO = 0.50              # probabilidade mínima para aprovar
SEED = 42                            # garante resultados reproduzíveis


# ---------------------------------------------------------------------------
# 2. DADOS SINTÉTICOS
# ---------------------------------------------------------------------------
def montar_features(idade, renda, parcela):
    """Transforma os dados brutos nas colunas que a rede recebe.

    Escalamos os valores (dividindo por um valor típico) para que fiquem
    próximos de 0-1; isso ajuda a rede neural a aprender mais rápido.
    A 4ª coluna (comprometimento = parcela / renda) é uma "feature
    engineering": entregamos a razão pronta, que é o que mais pesa no crédito.
    """
    idade = np.asarray(idade, dtype="float32")
    renda = np.asarray(renda, dtype="float32")
    parcela = np.asarray(parcela, dtype="float32")
    comprometimento = parcela / np.maximum(renda, 1.0)
    return np.column_stack(
        [idade / 100.0, renda / 20000.0, parcela / 5000.0, comprometimento]
    ).astype("float32")


def gerar_dados(n=20000):
    """Cria clientes fictícios e rotula cada um como aprovado (1) ou não (0)."""
    rng = np.random.default_rng(SEED)
    idade = rng.integers(16, 80, n)
    renda = rng.uniform(800, 15000, n)
    parcela = rng.uniform(100, 3000, n)

    aprovado = (
        (idade >= IDADE_MIN)
        & (idade <= IDADE_MAX)
        & (renda >= RENDA_MINIMA)
        & (parcela / renda <= COMPROMETIMENTO_MAX)
    ).astype("float32")
    return montar_features(idade, renda, parcela), aprovado


# ---------------------------------------------------------------------------
# 3. MODELO TENSORFLOW
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparando o modelo de análise de crédito...")
def carregar_modelo():
    """Treina a rede uma única vez; o cache evita retreinar a cada clique."""
    keras.utils.set_random_seed(SEED)
    X, y = gerar_dados()

    modelo = keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),  # saída: prob. 0 a 1
        ]
    )
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    modelo.fit(X, y, epochs=25, batch_size=128, validation_split=0.2, verbose=0)
    return modelo


def probabilidade(modelo, idade, renda, parcela):
    """Retorna a probabilidade (0 a 1) de aprovação para um cliente."""
    x = montar_features([idade], [renda], [parcela])
    return float(modelo.predict(x, verbose=0)[0][0])


def sugerir_parcela(modelo, idade, renda, parcela):
    """Reduz a parcela em passos de 2% até o modelo aprovar.

    Retorna None se o problema não for a parcela (ex.: idade fora da faixa
    ou renda abaixo do mínimo) - nesse caso, diminuir a parcela não resolve.
    """
    for fator in np.arange(0.98, 0.0, -0.02):
        candidata = round(parcela * fator, 2)
        if candidata < 50:
            break
        if probabilidade(modelo, idade, renda, candidata) >= LIMIAR_APROVACAO:
            return candidata
    return None


# ---------------------------------------------------------------------------
# 4. INTERFACE STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Análise de Crédito - Moto", page_icon="🏍️", layout="centered")

st.title("🏍️ Análise de Crédito para Financiamento de Moto")
st.caption("Ferramenta de uso interno · Simulação de pré-aprovação com rede neural (TensorFlow)")
st.divider()

modelo = carregar_modelo()

# st.form agrupa os campos: o cálculo só roda ao clicar no botão.
with st.form("form_credito"):
    st.subheader("Dados do cliente")
    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade (anos)", min_value=1, max_value=120, value=25, step=1)
        renda = st.number_input("Renda mensal (R$)", min_value=0.0, value=3000.0, step=100.0, format="%.2f")
    with col2:
        parcela = st.number_input("Valor da parcela desejada (R$)", min_value=0.0, value=600.0, step=50.0, format="%.2f")
    enviar = st.form_submit_button("Analisar crédito", use_container_width=True)

st.subheader("Resultado da análise")

if not enviar:
    st.info("Preencha os dados e clique em **Analisar crédito**.")
elif renda <= 0 or parcela <= 0:
    st.warning("Informe valores de renda e parcela maiores que zero.")
else:
    prob = probabilidade(modelo, idade, renda, parcela)
    comprometimento = parcela / renda

    if prob >= LIMIAR_APROVACAO:
        st.success("✅ **Foi aprovado**")
    else:
        st.error("❌ **Não aprovado**")

        sugestao = sugerir_parcela(modelo, idade, renda, parcela)
        if sugestao is not None:
            st.warning(f"💡 Sugestão: uma parcela de até **R$ {sugestao:,.2f}** tem chance de aprovação.")
        else:
            st.warning(
                "Reduzir a parcela não é suficiente para este perfil "
                "(idade fora da faixa aceita ou renda abaixo do mínimo)."
            )

    m1, m2 = st.columns(2)
    m1.metric("Confiança de aprovação", f"{prob:.0%}")
    m2.metric("Comprometimento da renda", f"{comprometimento:.0%}")

with st.expander("Política de crédito utilizada (simulação)"):
    st.markdown(
        f"- Idade entre **{IDADE_MIN}** e **{IDADE_MAX}** anos\n"
        f"- Renda mensal mínima de **R$ {RENDA_MINIMA:,.2f}**\n"
        f"- Parcela de até **{COMPROMETIMENTO_MAX:.0%}** da renda\n\n"
        "O modelo foi treinado com dados sintéticos para fins de demonstração; "
        "não substitui a análise oficial de crédito."
    )
