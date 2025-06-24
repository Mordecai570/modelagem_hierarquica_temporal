import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.stattools import pacf
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import pandas as pd


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

def boxplot_consumo_mensal(df):
    # Garantir que a coluna de data está no tipo datetime
    df["din_instante"] = pd.to_datetime(df["din_instante"])

    # Criar coluna com nome do mês (abreviado) e garantir ordenação correta
    df["mes"] = df["din_instante"].dt.month
    df["nome_mes"] = df["din_instante"].dt.strftime("%b")
    ordem_meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df["nome_mes"] = pd.Categorical(df["nome_mes"], categories=ordem_meses, ordered=True)

    # Criar boxplot com Plotly
    fig = px.box(
        df,
        x="nome_mes",
        y="val_cargaenergiamwmed",
        points="all",  # mostra todos os pontos no gráfico
        color="nome_mes",
        title="Distribuição Mensal do Consumo de Energia",
        labels={
            "nome_mes": "Mês",
            "val_cargaenergiamwmed": "Carga Energética (MW méd)"
        }
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_white"
    )

    return st.plotly_chart(fig)


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


def rolling_sarima_forecast(df, date_col, value_col,
                            order, seasonal_order,
                            h=1, num_rolling=90, window_length=730, freq="D"):
    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col).asfreq(freq)

    last_date = df.index.max()
    results = {"forecast_date": [], "forecast_value": []}

    # janela móvel: o primeiro end_train termina em last_date - (num_rolling-1)
    for i in range(num_rolling):
        end_train = last_date - timedelta(days=(num_rolling - 1 - i))
        start_train = end_train - timedelta(days=window_length - 1)
        train = df.loc[start_train:end_train, value_col]

        # data que será prevista
        fc_date = end_train + timedelta(days=h)

        if len(train) < window_length:
            results["forecast_date"].append(fc_date)
            results["forecast_value"].append(np.nan)
            continue

        try:
            model = SARIMAX(train,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            res = model.fit(disp=False)
            fc = res.forecast(steps=h)
            results["forecast_date"].append(fc_date)
            results["forecast_value"].append(fc.iloc[-1])
        except:
            results["forecast_date"].append(fc_date)
            results["forecast_value"].append(np.nan)

    return pd.DataFrame(results)


# --- Função para plotar SARIMA vs Real ---
def plot_forecast_vs_observed(df_forecast, df_real,
                              date_col="forecast_date",
                              forecast_col="forecast_value",
                              real_col="real"):
    df = df_forecast.copy()
    df = df.merge(df_real, left_on=date_col, right_on="din_instante", how="left")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[forecast_col],
        mode="lines",
        name="Previsão SARIMA",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[real_col],
        mode="lines",
        name="Valor Real",
        line=dict(color="black", dash="dash")
    ))

    fig.update_layout(
        title="SARIMA - Previsão vs Real",
        xaxis_title="Data",
        yaxis_title="Carga de Energia (MWméd)",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2)
    )

    return fig


sec = st.sidebar.selectbox(
    " ",
    ("Objetivo", "Análise Exploratória", "Modelo por Região", "Modelo Hierárquico"))


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
        boxplot_consumo_mensal(df_norte)
        st.plotly_chart(corr)
    elif reg_eda in "Nordeste":
        corr = correlogramas(df_nordeste)
        st.plotly_chart(linha_nordeste)
        boxplot_consumo_mensal(df_nordeste)
        st.plotly_chart(corr)
    elif reg_eda in "Sudeste/Centro-oeste":
        corr = correlogramas(df_sudeste_centroeste)
        st.plotly_chart(linha_sudeste_centroeste)
        boxplot_consumo_mensal(df_sudeste_centroeste)
        st.plotly_chart(corr)
    elif reg_eda in "Sul":
        corr = correlogramas(df_sul)
        st.plotly_chart(linha_sul)
        boxplot_consumo_mensal(df_sul)
        st.plotly_chart(corr)



if sec == "Modelo por Região":
    st.title("SARIMA por Região")

    # 1️⃣ Seleção de região
    reg_sarima = st.radio(
        "Selecione uma região",
        ["Norte", "Nordeste", "Sudeste/Centro-Oeste", "Sul"],
        horizontal=True
    )
    st.divider()
    # 2️⃣ Carrega e prepara a série
    df_sel = {
        "Norte": df_norte,
        "Nordeste": df_nordeste,
        "Sudeste/Centro-Oeste": df_sudeste_centroeste,
        "Sul": df_sul
    }[reg_sarima].copy()

    df_sel["din_instante"] = pd.to_datetime(df_sel["din_instante"])
    df_sel.set_index("din_instante", inplace=True)
    df_sel = df_sel.asfreq("D")

    serie = df_sel["val_cargaenergiamwmed"]
    analise = st.selectbox(" ", ["Critérios de informação", "Resíduos", "Previsão"])
    st.divider()
    if analise not in "Previsão":
        if reg_sarima in "Sul" or reg_sarima in "Sudeste/Centro-Oeste":
            st.write("**Recomendação: ARIMA(0,0,4)(0,1,1)[7]**")
        elif reg_sarima in "Nordeste":
            st.write("**Recomendação: ARIMA(2,0,0)(0,1,1)[7]**")
        elif reg_sarima in "Norte":
            st.write("**Recomendação: ARIMA(4,0,0)(0,1,1)[7]**")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("p (AR)", min_value=0, max_value=10, value=1, step=1)
            p1 = st.number_input("P (SAR)", min_value=0, max_value=10, value=0, step=1)
        with col2:
            d = st.number_input("d (diferença)", min_value=0, max_value=5, value=0, step=1)
            d1 = st.number_input("D (sazonal)", min_value=0, max_value=5, value=1, step=1)
        with col3:
            q = st.number_input("q (MA)", min_value=0, max_value=10, value=0, step=1)
            q1 = st.number_input("Q (SMA)", min_value=0, max_value=10, value=1, step=1)
        st.divider()
        try:
            # 3️⃣ Ajuste do SARIMA(2,0,0)(0,1,1)[7]
            modelo = SARIMAX(
                serie,
                order=(p, d, q),
                seasonal_order=(p1, d1, q1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True
            )
            resultado = modelo.fit(disp=False)

            if analise in "Critérios de informação":

                aic = resultado.aic
                bic = resultado.bic
                k = resultado.params.shape[0]  # número de parâmetros
                n = int(resultado.nobs)  # número de observações
                aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

                col1, col2, col3 = st.columns(3)

                col1.metric(label="AIC", value=f"{aic:.2f}")
                col2.metric(label="BIC", value=f"{bic:.2f}")
                col3.metric(label="AICc", value=f"{aicc:.2f}")

            if analise in "Resíduos":

                residuos = resultado.resid
                fig_resid = px.line(x=residuos.index, y=residuos, title="Resíduos ao longo do tempo")
                fig_resid.update_layout(
                    xaxis_title="Data",
                    yaxis_title="Resíduo",
                    template="plotly_white"
                )
                fig_resid.add_shape(type="line", x0=residuos.index.min(), x1=residuos.index.max(),
                                    y0=0, y1=0, line=dict(color="gray", dash="dash"))
                st.subheader("Resíduos no tempo")
                st.plotly_chart(fig_resid)

                lags = 40
                acf_vals = acf(residuos, nlags=lags)
                conf_int = 1.96 / np.sqrt(len(residuos))

                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(x=list(range(lags + 1)), y=acf_vals, name="ACF", marker_color="indianred"))
                fig_acf.add_trace(go.Scatter(x=list(range(lags + 1)), y=[conf_int] * (lags + 1), line=dict(dash="dot", color="gray")))
                fig_acf.add_trace(go.Scatter(x=list(range(lags + 1)), y=[-conf_int] * (lags + 1), line=dict(dash="dot", color="gray")))
                fig_acf.update_layout(title="Autocorrelação dos Resíduos (ACF)", xaxis_title="Defasagem (Lag)", yaxis_title="Autocorrelação", template="plotly_white", showlegend=False)
                st.subheader("ACF dos Resíduos")
                st.plotly_chart(fig_acf)

                pacf_vals = pacf(residuos, nlags=lags)
                fig_pacf = go.Figure()
                fig_pacf.add_trace(go.Bar(x=list(range(lags + 1)), y=pacf_vals, name="PACF", marker_color="steelblue"))
                fig_pacf.add_trace(go.Scatter(x=list(range(lags + 1)), y=[conf_int] * (lags + 1), line=dict(dash="dot", color="gray")))
                fig_pacf.add_trace(go.Scatter(x=list(range(lags + 1)), y=[-conf_int] * (lags + 1), line=dict(dash="dot", color="gray")))
                fig_pacf.update_layout(title="Autocorrelação Parcial dos Resíduos (PACF)", xaxis_title="Defasagem (Lag)", yaxis_title="PACF", template="plotly_white", showlegend=False)
                st.subheader("PACF dos Resíduos")
                st.plotly_chart(fig_pacf)



        except Exception as e:
            st.error(f"Erro ao ajustar modelo SARIMA: {e}")

    else:
        # 1️⃣ Carrega previsões prontas do GitHub
        url_fc = (
            "https://raw.githubusercontent.com/"
            "abibernardo/modelagem_hierarquica_temporal/"
            "main/previsoes_brasil.csv"
        )
        df_fc = pd.read_csv(url_fc, parse_dates=["forecast_date"])

        # 2️⃣ Mapeia coluna de previsão para cada região
        col_map = {
            "Norte": "sarima_norte",
            "Nordeste": "sarima_nordeste",
            "Sul": "sarima_sul",
            "Sudeste/Centro-Oeste": "sarima_sudeste_centroeste"
        }
        col_fc = col_map[reg_sarima]

        df_pred = df_fc[["forecast_date", col_fc]].rename(columns={col_fc: "forecast"})

        # 3️⃣ Carrega dados reais de 2025 do GitHub
        url_real = (
            "https://raw.githubusercontent.com/abibernardo/"
            "modelagem_hierarquica_temporal/main/"
            "CARGA_ENERGIA_2025%20(3).csv"
        )
        df_real = (
            pd.read_csv(url_real, sep=";", decimal=".")
                .assign(date=lambda d: pd.to_datetime(d["din_instante"]))
                .query("nom_subsistema == @reg_sarima")
                .groupby("date", as_index=False)["val_cargaenergiamwmed"]
                .sum()
                .rename(columns={"val_cargaenergiamwmed": "real"})
        )

        # 4️⃣ Faz merge só por data
        df_plot = pd.merge(
            df_pred,
            df_real,
            left_on="forecast_date",
            right_on="date",
            how="inner"
        )

        # 5️⃣ Desenha o gráfico
        fig = go.Figure([
            go.Scatter(
                x=df_plot["forecast_date"], y=df_plot["forecast"],
                mode="lines", name="Predito", line=dict(color="orange")
            ),
            go.Scatter(
                x=df_plot["forecast_date"], y=df_plot["real"],
                mode="lines", name="Real", line=dict(color="black", dash="dash")
            )
        ])
        fig.update_layout(
            title=f"{reg_sarima} — Previsão SARIMA vs Observado 2025",
            xaxis_title="Data",
            yaxis_title="Carga (MW méd)",
            template="plotly_white",
            legend=dict(orientation="h", y=-0.2)
        )
        fig.update_yaxes(rangemode="tozero")

        st.plotly_chart(fig, use_container_width=True)

if sec in "Modelo Hierárquico":
    st.title("Modelo Hierárquico: Bottom-Up")

    # 1️⃣ Multiselect para regiões
    regioes = ["Norte", "Nordeste", "Sul", "Sudeste/Centro-oeste"]
    selecionadas = st.multiselect("Selecione as regiões a agregar", regioes, default=regioes)

    # 2️⃣ Carrega previsões do GitHub
    url_fc = (
        "https://raw.githubusercontent.com/"
        "abibernardo/modelagem_hierarquica_temporal/"
        "main/previsoes_brasil.csv"
    )
    df_fc = pd.read_csv(url_fc, parse_dates=["forecast_date"])

    # 3️⃣ Mapeamento das colunas
    col_map = {
        "Norte": "sarima_norte",
        "Nordeste": "sarima_nordeste",
        "Sul": "sarima_sul",
        "Sudeste/Centro-oeste": "sarima_sudeste_centroeste"
    }

    # 4️⃣ Soma previsões selecionadas
    colunas_somar = [col_map[r] for r in selecionadas]
    df_fc["previsao_agregada"] = df_fc[colunas_somar].sum(axis=1)

    # 5️⃣ Carrega dados reais totais do Brasil
    url_real = (
        "https://raw.githubusercontent.com/abibernardo/"
        "modelagem_hierarquica_temporal/main/"
        "CARGA_ENERGIA_2025%20(3).csv"
    )
    df_real = pd.read_csv(url_real, sep=";", decimal=".")
    df_real["din_instante"] = pd.to_datetime(df_real["din_instante"])

    df_real = (
        df_real
            .groupby("din_instante", as_index=False)["val_cargaenergiamwmed"]
            .sum()
            .rename(columns={"din_instante": "data", "val_cargaenergiamwmed": "real_brasil"})
    )

    # 6️⃣ Merge com previsões
    df_plot = pd.merge(
        df_fc[["forecast_date", "previsao_agregada"]],
        df_real,
        left_on="forecast_date",
        right_on="data",
        how="inner"
    )

    # 7️⃣ Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot["forecast_date"], y=df_plot["previsao_agregada"],
        mode="lines", name="Soma das Previsões Regionais", line=dict(color="royalblue")
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["forecast_date"], y=df_plot["real_brasil"],
        mode="lines", name="Valor Real Brasil", line=dict(color="black", dash="dash")
    ))
    fig.update_layout(
        title="Agregação das Previsões Regionais vs Valor Real Brasil (2025)",
        xaxis_title="Data",
        yaxis_title="Carga Energética (MW méd)",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2)
    )
    fig.update_yaxes(rangemode="tozero")

    st.plotly_chart(fig, use_container_width=True)
