# ===============================
# PACOTES
# ===============================
pacotes <- c(
  "dplyr", "readr", "ggplot2", "plotly", "fpp3","stringr",
  "forecast", "prophet", "ggfortify", "broom", "lubridate", "tsibble", "fable", "fabletools", "feasts"
)

# Carrega os pacotes
lapply(pacotes, library, character.only = TRUE)


# ===============================
# 1. CARREGAR DADOS
# ===============================
url <- "https://raw.githubusercontent.com/abibernardo/modelagem_hierarquica_temporal/refs/heads/main/dados_me607.csv"
df_total <- read_csv(url) |> arrange(din_instante)

# ===============================
# 2. SERIE TOTAL DO BRASIL
# ===============================
df_brasil <- df_total |>
  group_by(din_instante) |>
  summarise(val_cargaenergiamwmed = sum(val_cargaenergiamwmed))

colnames(df_brasil) <- c("dia","carga")

df_brasil$dia <- str_replace_all(df_brasil$dia, "T00:00:00.000", "")

df_brasil$dia <- as.Date(df_brasil$dia) 

df_brasil$carga <- as.numeric(df_brasil$carga)

df_brasil <- df_brasil %>% as_tsibble(index = dia)

# ===============================
# 2. TABELA DOS MODELOS
# ===============================

modelos <- df_brasil %>% model(media = MEAN(carga),
                              naive = NAIVE(carga),
                              snaive = SNAIVE(carga ~ lag(7)),
                              drift = RW(carga ~ drift()),
                              ETS = ETS(carga ~ error("A") + trend("Ad") + season("A")),
                              ETS_multi = ETS(carga ~ error("M") + trend("M") + season("M")),
                              TSLM = TSLM(carga ~ trend() + fourier(K = 2)),
                              AR = ARIMA(carga ~ pdq(1,0,0) + PDQ(0,0,0)),
                              AR2 = ARIMA(carga ~ pdq(2,0,0) + PDQ(0,0,0)),
                              MA = ARIMA(carga ~ pdq(0,0,01) + PDQ(0,0,0)),
                              MA2 = ARIMA(carga ~ pdq(0,0,2) + PDQ(0,0,0)),
                              ARMA11 = ARIMA(carga ~ pdq(1,0,1) + PDQ(0,0,0)),
                              ARMA21 = ARIMA(carga ~ pdq(2,0,1) + PDQ(0,0,0)),
                              ARIMA111 = ARIMA(carga ~ pdq(1,1,1) + PDQ(0,0,0)),
                              ARIMA211 = ARIMA(carga ~ pdq(2,1,1) + PDQ(0,0,0)),
                              ARIMA112 = ARIMA(carga ~ pdq(1,1,2) + PDQ(0,0,0)),
                              SARIMA111 = ARIMA(carga ~ pdq(1,1,1) + PDQ(1,1,1)),
                              SARIMA211 = ARIMA(carga ~ pdq(2,1,1) + PDQ(2,1,1)),
                              SARIMA112 = ARIMA(carga ~ pdq(1,1,2) + PDQ(1,1,2)))

tab_acuracia <- accuracy(modelos)

min(tab_acuracia$ME)
min(tab_acuracia$RMSE)
min(tab_acuracia$MAE)
min(tab_acuracia$MPE)
min(tab_acuracia$MAPE)
min(tab_acuracia$MASE)
min(tab_acuracia$RMSSE)

#o modelo SARIMA(1,1,2) em geral saem melhor 
