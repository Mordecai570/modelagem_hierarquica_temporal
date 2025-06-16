# ===============================
# PACOTES
# ===============================
pacotes <- c(
  "dplyr", "tidyr", "readr", "ggplot2", "plotly",
  "forecast", "gridExtra", "prophet", "ggfortify", "broom", "lubridate", "tsibble", "fable", "fabletools", "feasts", "gridExtra"
)
# In
stala pacotes faltantes
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


#library(zoo)
library(forecast)

### Forecast BOTTOM-UP "na mão" (somando as previsões de cada região para a previsão nacional)

h <- 1
num_rolling <- 90
window_length <- 365
first_forecast_date <- as.Date("2025-01-01")

rolling_sarima_future <- function(df, h = 1, num_rolling = 90, window_length = 365) {
  # Série temporal como zoo
  inflation <- zoo(df$val_cargaenergiamwmed, order.by = df$din_instante)
  
  # Corrigido: garantir que seja Date
  last_date <- as.Date(max(index(inflation)))
  
  results <- data.frame(
    step = 1:num_rolling,
    forecast_date = seq.Date(from = last_date + 1, by = "day", length.out = num_rolling),
    forecast_value_sarima = NA_real_
  )
  
  for (i in 1:num_rolling) {
    # Janela de treino se move: termina em last_date - num_rolling + i
    est_end_date <- last_date - num_rolling + i
    est_start_date <- est_end_date - window_length + 1
    
    train_series <- window(inflation, start = est_start_date, end = est_end_date)
    
    if (length(train_series) < window_length) break
    
    train_ts <- ts(as.numeric(train_series), frequency = 7)
    
    fit <- Arima(train_ts, order = c(1, 0, 0), seasonal = c(2, 1, 2))
    fc <- forecast(fit, h = h)
    
    results$forecast_value_sarima[i] <- fc$mean[h]
  }
  
  return(results)
}

prev_nordeste <- rolling_sarima_future(df_nordeste)
prev_norte <- rolling_sarima_future(df_norte)
prev_sul <- rolling_sarima_future(df_sul)
prev_sudeste_centroeste <- rolling_sarima_future(df_sudeste_centroeste)

previsoes_brasil <- Reduce(function(x, y) {
  merge(x, y, by = "forecast_date", all = TRUE)
}, list(
  prev_nordeste[, c("forecast_date", "forecast_value_sarima")],
  prev_norte[, c("forecast_date", "forecast_value_sarima")],
  prev_sul[, c("forecast_date", "forecast_value_sarima")],
  prev_sudeste_centroeste[, c("forecast_date", "forecast_value_sarima")]
))

names(previsoes_brasil) <- c(
  "forecast_date",
  "sarima_nordeste",
  "sarima_norte",
  "sarima_sul",
  "sarima_sudeste_centroeste"
)

previsoes_brasil$forecast_value_total <- rowSums(
  previsoes_brasil[, 2:5], na.rm = TRUE
)


# 1. Filtrar observados do Brasil a partir de 2024
dados_obs <- df_brasil %>%
  filter(din_instante >= as.Date("2024-01-01")) %>%
  select(data = din_instante, carga = val_cargaenergiamwmed) %>%
  mutate(tipo = "Observado")

# 2. Previsões já somadas
dados_prev <- previsoes_brasil %>%
  select(data = forecast_date, carga = forecast_value_total) %>%
  mutate(tipo = "Previsto")

# 3. Juntar tudo
df_plot <- bind_rows(dados_obs, dados_prev)

# 4. Gráfico
ggplot(df_plot, aes(x = data, y = carga, color = tipo)) +
  geom_line(size = 1) +
  labs(
    title = "Carga de Energia no Brasil — Observado (2024) vs Previsto (2025)",
    x = "Data", y = "Carga (MW méd)",
    color = "Tipo"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("Observado" = "black", "Previsto" = "blue"))
