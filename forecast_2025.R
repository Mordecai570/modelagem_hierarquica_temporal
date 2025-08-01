# ===============================
# PACOTES
# ===============================
pacotes <- c(
  "dplyr", "Metrics", "tidyr", "readr", "ggplot2", "plotly",
  "forecast", "gridExtra", "prophet", "ggfortify", "broom", "lubridate", "tsibble", "fable", "fabletools", "feasts", "gridExtra"
)
# Instala pacotes faltantes
pacotes_faltando <- pacotes[!(pacotes %in% installed.packages()[, "Package"])]
if(length(pacotes_faltando)) install.packages(pacotes_faltando)

# Carrega pacotes
lapply(pacotes, library, character.only = TRUE)
library(zoo)
library(forecast)


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







########## verificação de qual sarima é melhor: 

# Função para ajustar auto.arima em uma série regional
ajustar_auto_arima <- function(df) {
  df <- df |> arrange(din_instante)
  serie <- zoo(df$val_cargaenergiamwmed, order.by = df$din_instante)
  ts_serie <- ts(as.numeric(serie), frequency = 7)  # frequência semanal
  
  modelo <- auto.arima(
    ts_serie,
    seasonal = TRUE,
    stepwise = FALSE,         # busca exaustiva
    approximation = FALSE,    # mais preciso (mais lento)
    lambda = "auto"           # Box-Cox se necessário
  )
  
  return(modelo)
}

# Aplicar para cada região
modelo_nordeste <- ajustar_auto_arima(df_nordeste)
modelo_norte <- ajustar_auto_arima(df_norte)
modelo_sul <- ajustar_auto_arima(df_sul)
modelo_sudeste_centroeste <- ajustar_auto_arima(df_sudeste_centroeste)
modelo_brasil <- ajustar_auto_arima(df_brasil)

# Exibir resultados
modelo_nordeste
modelo_norte
modelo_sul
modelo_sudeste_centroeste
modelo_brasil


### Fazendo previsão hierárquica


h <- 1
num_rolling <- 90
window_length <- 730
first_forecast_date <- as.Date("2025-01-01")

rolling_sarima_nordeste <- function(df, h = 1, num_rolling = 90, window_length = 730) {
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
    
    fit <- Arima(train_ts, order = c(2, 0, 0), seasonal = c(0, 1, 1))
    fc <- forecast(fit, h = h)
    
    results$forecast_value_sarima[i] <- fc$mean[h]
  }
  
  return(results)
}


rolling_sarima_norte <- function(df, h = 1, num_rolling = 90, window_length = 730) {
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
    
    fit <- Arima(train_ts, order = c(4, 0, 0), seasonal = c(0, 1, 1))
    fc <- forecast(fit, h = h)
    
    results$forecast_value_sarima[i] <- fc$mean[h]
  }
  
  return(results)
}


rolling_sarima_sul <- function(df, h = 1, num_rolling = 90, window_length = 730) {
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
    
    fit <- Arima(train_ts, order = c(0, 0, 4), seasonal = c(0, 1, 1))
    fc <- forecast(fit, h = h)
    
    results$forecast_value_sarima[i] <- fc$mean[h]
  }
  
  return(results)
}











prev_nordeste <- rolling_sarima_nordeste(df_nordeste)
prev_norte <- rolling_sarima_norte(df_norte)
prev_sul <- rolling_sarima_sul(df_sul)
prev_sudeste_centroeste <- rolling_sarima_sul(df_sudeste_centroeste)
prev_brasil_direto<- rolling_sarima_sul(df_brasil)


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

ggplot(df_plot, aes(x = data, y = carga, color = tipo)) +
  geom_line(size = 1) +
  labs(
    title = "Observado (2024) vs Previsto (2025)",
    x = "Data", y = "Carga (MW méd)",
    color = ""
  ) +
  theme_minimal()


#########
url_dados_2025 <- "https://raw.githubusercontent.com/abibernardo/modelagem_hierarquica_temporal/main/CARGA_ENERGIA_2025%20(3).csv"

df_2025 <- read_delim(
  file = url_dados_2025,
  delim = ";",
  locale = locale(decimal_mark = ".")
)

df_2025 <- df_2025 %>%
  mutate(
    din_instante = as.Date(din_instante, format = "%Y-%m-%d")
  )



# 1. Agrupa os reais de 2025
df_2025_real <- df_2025 %>%
  group_by(din_instante) %>%
  summarise(real = sum(val_cargaenergiamwmed), .groups = "drop")


# 1. Previsões SARIMA Direto
df_sarima_direto <- prev_brasil_direto %>%
  select(forecast_date, previsao = forecast_value_sarima) %>%
  mutate(metodo = "SARIMA Direto")

# 2. Previsões Bottom-Up
df_sarima_bottomup <- previsoes_brasil %>%
  select(forecast_date, previsao = forecast_value_total) %>%
  mutate(metodo = "SARIMA Bottom-Up")

# 3. Juntar previsões
df_comp_long <- bind_rows(df_sarima_direto, df_sarima_bottomup) %>%
  left_join(df_2025_real, by = c("forecast_date" = "din_instante"))




ggplot(df_comp_long, aes(x = forecast_date, y = previsao, color = metodo)) +
  geom_line(linewidth = 0.9, alpha = 0.9) +
  geom_line(aes(y = real), color = "black", linetype = "dashed", linewidth = 0.8) +
  labs(
    title = "Previsões do Consumo de Energia em 2025 vs Observado",
    subtitle = "Comparação entre SARIMA Direto e SARIMA Bottom-Up",
    x = "Data", y = "Carga de Energia (MW méd)",
    color = "Método"
  ) +
  scale_color_manual(values = c(
    "SARIMA Direto" = "#1f77b4",        # azul
    "SARIMA Bottom-Up" = "#ff7f0e"      # laranja
  )) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11),
    axis.title = element_text(size = 12),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10),
    axis.text = element_text(size = 10),
    legend.position = "bottom"
  ) +
  expand_limits(y = 0)



library(dplyr)
library(Metrics)

# Supondo que df_comp_long já existe e tem colunas:
#   forecast_date, previsao, metodo, real

# Filtra apenas observações com valor real disponível
df_erros <- df_comp_long %>%
  filter(!is.na(real)) %>%
  group_by(metodo) %>%
  summarise(
    RMSE = rmse(real, previsao),
    MAE  = mae(real, previsao),
    MAPE = mape(real, previsao),
    .groups = "drop"
  )

# Exibe o resultado
print(df_erros)
####################################################
##################################################### PARA CADA REGIÃO

# 1) Extrair reais para o Nordeste
df_real_nordeste <- df_2025 %>%
  filter(nom_subsistema == "Nordeste") %>%
  select(data = din_instante, real = val_cargaenergiamwmed)

# 2) Preparar previsões do Nordeste
#    Supondo que prev_nordeste tem colunas: step, forecast_date, forecast_value_sarima
df_prev_nordeste <- prev_nordeste %>%
  select(data, previsao)

# 3) Juntar séries
df_plot_nordeste <- full_join(df_real_nordeste, df_prev_nordeste, by = "data")

# 4) Plot
ggplot(df_comp_long, aes(x = forecast_date, y = previsao, color = metodo)) +
  geom_line(linewidth = 0.9, alpha = 0.9) +
