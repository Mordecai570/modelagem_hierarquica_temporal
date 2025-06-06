# ===============================
# PACOTES
# ===============================
pacotes <- c(
  "dplyr", "readr", "ggplot2", "plotly",
  "forecast", "gridExtra", "prophet", "ggfortify", "broom", "lubridate", "tsibble", "fable", "fabletools", "feasts", "gridExtra"
)
# Instala pacotes faltantes
pacotes_faltando <- pacotes[!(pacotes %in% installed.packages()[, "Package"])]
if(length(pacotes_faltando)) install.packages(pacotes_faltando)

# Carrega pacotes
lapply(pacotes, library, character.only = TRUE)

# ===============================
# 1. CARREGAR DADOS
# ===============================
url <- "https://raw.githubusercontent.com/abibernardo/modelagem_hierarquica_temporal/refs/heads/main/dados_me607.csv"
df_total <- read_csv(url) |> arrange(din_instante)

# Série total do Brasil
# ===============================
df_brasil <- df_total |> 
  group_by(din_instante) |> 
  summarise(val_cargaenergiamwmed = sum(val_cargaenergiamwmed), .groups = "drop")

# Dados por região
# ===============================
regioes <- unique(df_total$nom_subsistema)
dfs_regioes <- lapply(regioes, function(reg) df_total |> filter(nom_subsistema == reg))
names(dfs_regioes) <- regioes
df_nordeste <- dfs_regioes[["Nordeste"]]
df_norte <- dfs_regioes[["Norte"]]
df_sul <- dfs_regioes[["Sul"]]
df_sudeste_centroeste <- dfs_regioes[["Sudeste/Centro-Oeste"]]


# ===============================
# 2. FUNÇÕES DE MODELAGEM E DIAGNÓSTICO
# ===============================

# Função de diagnóstico de resíduos (estilo gg_tsresiduals)
analisar_residuos <- function(residuos_df, titulo = "") {
  p1 <- ggplot(residuos_df, aes(x = data, y = residuo)) +
    geom_line() + geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste(titulo, "- Resíduos no tempo"), x = "Data", y = "Resíduo")
  p2 <- ggplot(residuos_df, aes(x = residuo)) +
    geom_histogram(bins = 30) +
    labs(title = paste(titulo, "- Histograma dos resíduos"), x = "Resíduo", y = "Frequência")
  p3 <- autoplot(Acf(residuos_df$residuo, plot = FALSE)) +
    labs(title = paste(titulo, "- ACF dos resíduos"))
  grid.arrange(p1, p2, p3, ncol = 1)
}

# 2.1 Modelo SARIMA
ajustar_sarima <- function(df_regiao) {
  df_ts <- df_regiao %>% mutate(din_instante = as.Date(din_instante)) %>%
    as_tsibble(index = din_instante)
  modelo <- df_ts %>%
    model(SARIMA = ARIMA(val_cargaenergiamwmed ~ pdq(0:2,0:1,0:2) + PDQ(0:2,0:1,0:2)))
  print(report(modelo))
  fc <- forecast(modelo, h = 30)
  print(autoplot(fc, df_ts) + labs(title = "SARIMA Previsão", y = "MW médios", x = "Data"))
  # resíduos
  res <- modelo %>% augment() %>% select(din_instante, .innov) %>%
    rename(data = din_instante, residuo = .innov)
  analisar_residuos(res, titulo = "SARIMA")
}

# 2.2 Modelo Prophet
# 2.2 Modelo Prophet
ajustar_prophet <- function(df_regiao) {
  df_p <- df_regiao %>%
    transmute(ds = as.Date(din_instante), y = val_cargaenergiamwmed)
  
  modelo <- prophet(df_p,
                    yearly.seasonality = TRUE,
                    weekly.seasonality = FALSE,
                    daily.seasonality = FALSE)
  
  # gera futuros e previsão
  future <- make_future_dataframe(modelo, periods = 30)
  fc <- predict(modelo, future) %>%
    mutate(ds = as.Date(ds))    # <--- força Date nos forecasts
  
  # gráfico real vs fitted
  p <- ggplot() +
    geom_line(data = df_p, aes(x = ds, y = y), alpha = 0.5) +
    geom_line(data = fc, aes(x = ds, y = yhat), color = "blue") +
    labs(title = "Prophet Previsão", x = "Data", y = "MW médios")
  print(p)
  
  # resíduos
  df_res <- df_p %>%
    left_join(fc %>% select(ds, yhat), by = "ds") %>%
    mutate(residuo = y - yhat) %>%
    rename(data = ds)
  analisar_residuos(df_res, titulo = "Prophet")
}

# 2.3 Modelo ETS (Holt-Winters)
ajustar_ets <- function(df_regiao) {
  df_ts <- df_regiao %>% mutate(din_instante = as.Date(din_instante)) %>%
    arrange(din_instante) %>% as_tsibble(index = din_instante)
  modelo <- df_ts %>% model(ETS = ETS(val_cargaenergiamwmed ~ error("A") + trend("Ad") + season("A")))
  print(report(modelo))
  fc <- forecast(modelo, h = 30)
  print(autoplot(fc, df_ts) + labs(title = "ETS Previsão", y = "MW médios", x = "Data"))
  # resíduos
  res <- modelo %>% augment() %>% select(din_instante, .innov) %>%
    rename(data = din_instante, residuo = .innov)
  analisar_residuos(res, titulo = "ETS")
}

# 2.4 Modelo TSLM (Trend + Fourier)
ajustar_tslm <- function(df_regiao, K = 2) {
  df_ts <- df_regiao %>% mutate(din_instante = as.Date(din_instante)) %>% as_tsibble(index = din_instante)
  modelo <- df_ts %>% model(TSLM = TSLM(val_cargaenergiamwmed ~ trend() + fourier(K = K)))
  print(report(modelo))
  fc <- forecast(modelo, h = 30)
  print(autoplot(fc, df_ts) + labs(title = "TSLM Previsão", y = "MW médios", x = "Data"))
  # resíduos com fabletools
  res <- modelo %>% augment() %>% select(din_instante, .innov) %>% rename(data = din_instante, residuo = .innov)
  analisar_residuos(res, titulo = "TSLM")
}

# ===============================
# 3. APLICAR FUNÇÕES EM CADA REGIÃO
# ===============================
# Região Nordeste
ajustar_sarima(df_nordeste)
ajustar_prophet(df_nordeste)
ajustar_ets(df_nordeste)
ajustar_tslm(df_nordeste, K = 2)

# Região Norte
ajustar_sarima(df_norte)
ajustar_prophet(df_norte)
ajustar_ets(df_norte)
ajustar_tslm(df_norte, K = 2)

# Região Sul
ajustar_sarima(df_sul)
ajustar_prophet(df_sul)
ajustar_ets(df_sul)
ajustar_tslm(df_sul, K = 2)

# Região Sudeste + Centro-Oeste
ajustar_sarima(df_sudeste_centroeste)
ajustar_prophet(df_sudeste_centroeste)
ajustar_ets(df_sudeste_centroeste)
ajustar_tslm(df_sudeste_centroeste, K = 2)

# Brasil (total)
ajustar_sarima(df_brasil)
ajustar_prophet(df_brasil)
ajustar_ets(df_brasil)
ajustar_tslm(df_brasil, K = 2)
