import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

# CARREGAR E DIVIDIR DATAFRAMES
url = "https://raw.githubusercontent.com/abibernardo/modelagem_hierarquica_temporal/refs/heads/main/dados_me607.csv"
df_total = pl.read_csv(url)
df_total = df_total.sort("din_instante")
regioes = df_total["nom_subsistema"].unique().to_list()

dfs_regioes = {regiao: df_total.filter(pl.col("nom_subsistema") == regiao) for regiao in regioes}
df_nordeste = dfs_regioes["Nordeste"]
df_sul = dfs_regioes["Sul"]
df_sudeste_centroeste = dfs_regioes["Sudeste/Centro-Oeste"]
df_norte = dfs_regioes["Norte"]
df_brasil = (df_total.group_by("din_instante").agg(pl.col("val_cargaenergiamwmed").sum().alias("val_cargaenergiamwmed")).sort("din_instante"))

df_total = df_total.to_pandas()
df_brasil = df_brasil.to_pandas()
df_nordeste = df_nordeste.to_pandas()
df_norte = df_norte.to_pandas()
df_sul = df_sul.to_pandas()
df_sudeste_centroeste = df_sudeste_centroeste.to_pandas()


#################################################
# GRÁFICOS DE LINHAS

linha_brasil = px.line(
    df_brasil,
    x="din_instante",
    y="val_cargaenergiamwmed",
    title="Consumo de Energia Diário no Brasil",
    labels={"din_instante": "Data", "val_cargaenergiamwmed": "Carga Energética (MW médio)", "nom_subsistema": "Região"}
)

linha_regiao = px.line(
    df_total,
    x="din_instante",
    y="val_cargaenergiamwmed",
    color="nom_subsistema",
    title="Consumo de Energia Diário por Região",
    labels={"din_instante": "Data", "val_cargaenergiamwmed": "Carga Energética (MW médio)", "nom_subsistema": "Região"}
)
linha_regiao.update_layout(xaxis_title="Data", yaxis_title="Carga Energética (MW médio)", xaxis_tickangle=-45)

linha_sul = px.line(
    df_sul,
    x="din_instante",
    y="val_cargaenergiamwmed",
    title="Consumo de Energia Diário no Sul",
    labels={"din_instante": "Data", "val_cargaenergiamwmed": "Carga Energética (MW médio)", "nom_subsistema": "Região"}
)

linha_sudeste_centroeste = px.line(
    df_sudeste_centroeste,
    x="din_instante",
    y="val_cargaenergiamwmed",
    title="Consumo de Energia Diário no Sudeste e Centro-oeste",
    labels={"din_instante": "Data", "val_cargaenergiamwmed": "Carga Energética (MW médio)", "nom_subsistema": "Região"}
)

linha_norte = px.line(
    df_norte,
    x="din_instante",
    y="val_cargaenergiamwmed",
    title="Consumo de Energia Diário no Norte",
    labels={"din_instante": "Data", "val_cargaenergiamwmed": "Carga Energética (MW médio)", "nom_subsistema": "Região"}
)

linha_nordeste = px.line(
    df_nordeste,
    x="din_instante",
    y="val_cargaenergiamwmed",
    title="Consumo de Energia Diário no Nordeste",
    labels={"din_instante": "Data", "val_cargaenergiamwmed": "Carga Energética (MW médio)", "nom_subsistema": "Região"}
)


# CORRELOGRAMAS

def correlogramas(df):


    # Calcular a ACF (autocorrelação) para 40 defasagens
    acf_vals = acf(df["val_cargaenergiamwmed"].to_list(), nlags=40)

    # Criar o gráfico de autocorrelação com Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"))

    # Configurar layout do gráfico
    fig.update_layout(
        title=f"Correlograma",
        xaxis_title="Defasagem (Lags)",
        yaxis_title="Autocorrelação",
        template="plotly_white")

    return fig



#### DECOMPOSIÇÃO MULTIPLICATIVA (mudar para aditiva?)

FREQ = 365  # Ajuste conforme PERÍODO

# Função para decompor e plotar a série temporal
def decompor_e_plotar(df):

    # Definir índice temporal
    df.set_index("din_instante", inplace=True)

    # Aplicar decomposição clássica
    decomposed = seasonal_decompose(df["val_cargaenergiamwmed"], model="multiplicative", period=FREQ)

    # --- Gráfico de linha
    fig_original = go.Figure()
    fig_original.add_trace(
        go.Scatter(x=df.index, y=df["val_cargaenergiamwmed"], mode="lines", name="Série Original"))
    fig_original.update_layout(title=f"Série Temporal", xaxis_title="Data", yaxis_title="Consumo de Energia",
                               template="plotly_white")

    # --- 1️⃣ Gráfico da Tendência ---
    fig_tendencia = go.Figure()
    fig_tendencia.add_trace(go.Scatter(x=df.index, y=decomposed.trend, mode="lines", name="Tendência"))
    fig_tendencia.update_layout(title=f"Tendência", xaxis_title="Data", yaxis_title="Tendência",
                                template="plotly_white")

    # --- 2️⃣ Gráfico da Sazonalidade ---
    fig_sazonalidade = go.Figure()
    fig_sazonalidade.add_trace(go.Scatter(x=df.index, y=decomposed.seasonal, mode="lines", name="Sazonalidade"))
    fig_sazonalidade.update_layout(title=f"Sazonalidade", xaxis_title="Data", yaxis_title="Sazonalidade",
                                   template="plotly_white")

    # --- 3️⃣ Gráfico do Resíduo ---
    fig_residuo = go.Figure()
    fig_residuo.add_trace(go.Scatter(x=df.index, y=decomposed.resid, mode="lines", name="Resíduo"))
    fig_residuo.update_layout(title=f"Resíduo", xaxis_title="Data", yaxis_title="Resíduo",
                              template="plotly_white")

    # Exibir os gráficos
    st.plotly_chart(fig_original)

    st.plotly_chart(fig_tendencia)

    st.plotly_chart(fig_sazonalidade)

    st.plotly_chart(fig_residuo)

### PROPHET
def aplicar_prophet(df, titulo):
    # Converter para Pandas e formatar para o Prophet
    df_pd = df
    df_pd = df_pd.rename(columns={"din_instante": "ds", "val_cargaenergiamwmed": "y"})

    # Criar e ajustar o modelo
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(df_pd)

    # Criar previsões para os próximos 365 dias
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # --- 1️⃣ Gráfico da Previsão ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pd["ds"], y=df_pd["y"], mode="lines", name="Série Original"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Previsão", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Limite Superior", line=dict(dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Limite Inferior", line=dict(dash="dot", color="gray")))
    fig.update_layout(title=f"Previsão - {titulo}", xaxis_title="Data", yaxis_title="Consumo de Energia", template="plotly_white")
    st.plotly_chart(fig)

    # --- 2️⃣ Componentes da Série ---
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines", name="Tendência", line=dict(color="blue")))
    fig_comp.update_layout(title=f"Tendência da Série - {titulo}", xaxis_title="Data", yaxis_title="Tendência", template="plotly_white")
    st.plotly_chart(fig_comp)

    # --- 3️⃣ Resíduos ---
    df_residuos = df_pd.copy()
    df_residuos["yhat"] = forecast["yhat"][: len(df_pd)]
    df_residuos["residuo"] = df_residuos["y"] - df_residuos["yhat"]

    # Resíduos ao longo do tempo
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=df_residuos["ds"], y=df_residuos["residuo"], mode="lines+markers", name="Resíduos"))
    fig_res.add_trace(go.Scatter(x=df_residuos["ds"], y=[0] * len(df_residuos), mode="lines", name="Linha Zero", line=dict(dash="dot", color="red")))
    fig_res.update_layout(title=f"Resíduos ao Longo do Tempo - {titulo}", xaxis_title="Data", yaxis_title="Erro (y - yhat)", template="plotly_white")
    st.plotly_chart(fig_res)

    # --- 4️⃣ Sazonalidade Anual ---
    fig_seasonality = go.Figure()
    fig_seasonality.add_trace(
        go.Scatter(x=forecast["ds"], y=forecast["yearly"], mode="lines", name="Sazonalidade Anual",
                   line=dict(color="green")))
    fig_seasonality.update_layout(title=f"Sazonalidade Anual - {titulo}", xaxis_title="Data",
                                  yaxis_title="Variação Anual", template="plotly_white")
    st.plotly_chart(fig_seasonality)

### Montagem de tela

sec = st.sidebar.selectbox(
    " ",
    ("Objetivo", "Análise Exploratória", "Decomposição Multiplicativa", "Prophet"))


if sec in "Objetivo":
    st.title("Modelagem hierárquica do consumo diário de energia no Brasil")
    st.divider()
    st.subheader("Origem dos dados")
    st.write("Os dados do consumo diário de energia foram extraídos do site oficial da ONS (Operador Nacional do Sistema Elétrico). O consumo é medido em MWmed, e é dividido por região (consumo diário no Norte, Nordeste, Sul e Sudeste-Centroeste). Foram extraídos os dados para todas as regiões, desde 2027")
    st.divider()
    col1, col2 = st.columns(2)
    st.plotly_chart(linha_brasil)
    st.plotly_chart(linha_regiao)
    st.subheader("Objetivo")
    st.write("O objetivo deste trabalho é criar um modelo hierárquico de séries temporais bottom-up, de forma que se construa modelos para cada uma das regiões, depois agregando-os e criando um modelo para o consumo de energia total do Brasil.")
    st.write("Posteriormente, será modelada diretamente a série do consumo de energia total do país, ignorando a granularidade dos dados. E então, serão comparados o modelo hierárquico e o modelo 'direto'.")

elif sec in "Análise Exploratória":
    reg_eda = st.radio("Selecione uma região", ["Norte", "Nordeste", "Sudeste/Centro-oeste", "Sul"], key='radio 1', horizontal=True)
    st.divider()
    if reg_eda in "Norte":
        corr = correlogramas(df_norte)
        st.plotly_chart(linha_norte)
        st.plotly_chart(corr)
    elif reg_eda in "Nordeste":
        corr = correlogramas(df_nordeste)
        st.plotly_chart(linha_nordeste)
        st.plotly_chart(corr)
    elif reg_eda in "Sudeste/Centro-oeste":
        corr = correlogramas(df_sudeste_centroeste)
        st.plotly_chart(linha_sudeste_centroeste)
        st.plotly_chart(corr)
    elif reg_eda in "Sul":
        corr = correlogramas(df_sul)
        st.plotly_chart(linha_sul)
        st.plotly_chart(corr)



elif sec in "Decomposição Multiplicativa":
    reg_decomp = st.radio("Selecione uma região", ["Norte", "Nordeste", "Sudeste/Centro-oeste", "Sul"], horizontal=True)
    st.divider()
    if reg_decomp in "Norte":
        decompor_e_plotar(df_norte)

    elif reg_decomp in "Nordeste":
        decompor_e_plotar(df_nordeste)

    elif reg_decomp in "Sudeste/Centro-oeste":
        decompor_e_plotar(df_sudeste_centroeste)

    elif reg_decomp in "Sul":
        decompor_e_plotar(df_sul)


elif sec in "Prophet":
    reg_pro = st.radio("Selecione uma região", ["Norte", "Nordeste", "Sudeste/Centro-oeste", "Sul"], key='radio pro', horizontal=True)

    # Aplicar Prophet em cada série
    if reg_pro in "Nordeste":
        aplicar_prophet(df_nordeste, "Nordeste")
    if reg_pro in "Sul":
        aplicar_prophet(df_sul, "Sul")
    if reg_pro in "Sudeste/Centro-oeste":
        aplicar_prophet(df_sudeste_centroeste, "Sudeste/Centro-Oeste")
    if reg_pro in "Norte":
        aplicar_prophet(df_norte, "Norte")
    if reg_pro in "Brasil":
        aplicar_prophet(df_brasil, "Brasil")
